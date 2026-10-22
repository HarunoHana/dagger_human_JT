import numpy as np
from scipy.spatial.transform import Rotation

from .motion_variation import choose_motion


class ScriptedPickPlace:
    # Adjust these only after observing the first test.
    FINGER_CLOSE_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])
    LYING_GRASP_Z_OFFSET = -0.005
    UPRIGHT_GRASP_Z_OFFSET = 0.020

    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.stage = 0
        self.steps = 0
        self.retries = 0

        self.home_rotation = None
        self.grasp_position = None
        self.grasp_can_height = None
        self.grasp_distance = None
        self.lift_start = None
        self.locked_rotation = None

        self.debug_count = 0

        # One style per rollout (not one random choice per action).
        self.motion = choose_motion()
        self.approach_waypoint_reached = False
        self.transport_waypoint_reached = False
        self.transport_waypoint = None
        self.recovery_mode = False
        self.stage0_timeout = 110
        self.stage5_waypoint_timeout = 110
        print(f"MOTION: {self.motion.name}")

    def reset_rollout(self):
        self.reset()

    def observe(self):
        raw = self.env._last_raw_obs

        hand = np.asarray(raw["robot0_eef_pos"]).copy()
        can = np.asarray(raw["Can_pos"]).copy()

        hand_rot = Rotation.from_quat(
            np.asarray(raw["robot0_eef_quat"])
        )

        can_rot = Rotation.from_quat(
            np.asarray(raw["Can_quat"])
        )

        can_axis = can_rot.apply([0, 0, 1])
        tilt = np.degrees(
            np.arccos(np.clip(abs(can_axis[2]), 0, 1))
        )

        return hand, can, hand_rot, can_axis, tilt

    def change_stage(self, stage, message=""):
        self.stage = stage
        self.steps = 0

        if message:
            print(f"STAGE {stage}: {message}")

    def target_bin(self):
        base = self.env.env

        return np.asarray(
            base.target_bin_placements[
                base.object_to_id["can"]
            ]
        ).copy()

    def contact_grasp(self):
        """
        Return True/False if robosuite's grasp checker
        is available. Otherwise return None.
        """
        base = self.env.env

        try:
            robot = base.robots[0]
            gripper = robot.gripper

            can_object = next(
                obj for obj in base.objects
                if obj.name.lower() == "can"
            )

            return bool(
                base._check_grasp(
                    gripper=gripper,
                    object_geoms=can_object.contact_geoms
                )
            )
        except (AttributeError, TypeError, StopIteration):
            return None

    def desired_rotation(self, can_axis, tilt):
        """
        Maintain a top-down gripper pose.
        For a lying can, rotate about world Z so that
        the finger closing direction crosses the can.
        """
        base_rot = self.home_rotation

        if tilt < 35:
            return base_rot

        horizontal = np.array([
            can_axis[0],
            can_axis[1],
            0.0
        ])

        norm = np.linalg.norm(horizontal)

        if norm < 1e-6:
            return base_rot

        horizontal /= norm

        # Fingers should close ACROSS the cylinder,
        # not along its long axis.
        desired_close = np.array([
            -horizontal[1],
            horizontal[0],
            0.0
        ])

        current_close = base_rot.apply(
            self.FINGER_CLOSE_AXIS_LOCAL
        )

        current_close[2] = 0.0
        close_norm = np.linalg.norm(current_close)

        if close_norm < 1e-6:
            return base_rot

        current_close /= close_norm

        # A gripper's closing direction is bidirectional.
        # Choose the smaller wrist rotation.
        if np.dot(current_close, desired_close) < 0:
            desired_close = -desired_close

        cross_z = np.cross(
            current_close, desired_close
        )[2]

        dot = np.clip(
            np.dot(current_close, desired_close),
            -1.0,
            1.0
        )

        yaw = np.arctan2(cross_z, dot)

        return (
            Rotation.from_euler("z", yaw)
            * base_rot
        )

    def retry(self, reason):
        self.retries += 1
        print(
            f"RETRY {self.retries}: {reason}"
        )

        self.grasp_position = None
        self.grasp_can_height = None
        self.grasp_distance = None
        self.lift_start = None
        # On retries, skip decorative waypoints and use the tested direct route.
        self.recovery_mode = True
        self.approach_waypoint_reached = True
        self.transport_waypoint_reached = True
        self.transport_waypoint = None

        # Do not overwrite the original wrist pose.
        self.change_stage(0, "Replanning grasp")

    def resume_from_human(self):
        # Human may place robot in an unreachable pose for decorative waypoints.
        # Resume via the original direct route instead of forcing side detours.
        self.recovery_mode = True
        self.approach_waypoint_reached = True
        self.transport_waypoint_reached = True
        self.transport_waypoint = None
        hand, can, hand_rot, axis, tilt = self.observe()

        contact = self.contact_grasp()

        if contact is True:
            # Preserve the human's current wrist orientation during lift.
            self.locked_rotation = hand_rot
            # Check lifting before transporting.
            self.grasp_position = hand.copy()
            self.grasp_can_height = float(can[2])
            self.grasp_distance = float(
                np.linalg.norm(hand - can)
            )
            self.change_stage(
                4, "Human returned holding can"
            )

        else:
            self.grasp_position = None
            self.grasp_can_height = None
            self.grasp_distance = None

            self.change_stage(
                0, "Human returned; reacquiring can"
            )

    def act(self, obs=None):
        hand, can, hand_rot, can_axis, tilt = self.observe()

        if self.home_rotation is None:
            self.home_rotation = hand_rot

        self.steps += 1
        self.debug_count += 1

        distance = np.linalg.norm(hand - can)
        bin_pos = self.target_bin()

        action = np.zeros(7)
        gripper = -1.0
        target = hand.copy()

        desired_rot = self.desired_rotation(
            can_axis, tilt
        )

        # During transportation, preserve the grasp pose.
        if self.stage >= 4 and self.locked_rotation is not None:
            desired_rot = self.locked_rotation

        rotation_error = (
            desired_rot * hand_rot.inv()
        ).as_rotvec()

        angle_error = np.linalg.norm(rotation_error)

        # -------------------------------
        # 0: Position above the can
        # -------------------------------
        if self.stage == 0:
            gripper = -1.0

            # A safe lateral waypoint ABOVE the can; final pose unchanged.
            final_target = can + np.array([0.0, 0.0, 0.16])
            offset = np.asarray(self.motion.approach_offset_xy)
            if (
                np.any(offset)
                and not self.recovery_mode
                and not self.approach_waypoint_reached
            ):
                target = final_target + np.array([*offset, 0.03])
                if np.linalg.norm(target - hand) < 0.030:
                    self.approach_waypoint_reached = True
                elif self.steps >= self.stage0_timeout:
                    print("MOTION: approach waypoint timeout -> direct approach")
                    self.approach_waypoint_reached = True
            else:
                target = final_target
                position_ready = np.linalg.norm(target - hand) < 0.025
                rotation_ready = angle_error < 0.10
                if position_ready and rotation_ready:
                    self.change_stage(1, "Aligned above can")
                elif self.steps % 120 == 0:
                    print(
                        f"APPROACH: pos_error={np.linalg.norm(target - hand):.3f} "
                        f"rot_error_deg={np.degrees(angle_error):.1f}"
                    )

        # -------------------------------
        # 1: Lower with gripper open
        # -------------------------------
        elif self.stage == 1:
            gripper = -1.0

            if tilt >= 60:
                offset = self.LYING_GRASP_Z_OFFSET
            else:
                offset = self.UPRIGHT_GRASP_Z_OFFSET

            target = can + np.array([
                0.0, 0.0, offset
            ])

            position_ready = (
                np.linalg.norm(target - hand) < 0.020
            )

            rotation_ready = angle_error < 0.10

            if position_ready and rotation_ready:
                self.change_stage(
                    2, "Closing gripper"
                )

            elif self.steps > 300:
                self.retry(
                    "Could not reach grasp pose"
                )

        # -------------------------------
        # 2: Close the gripper
        # -------------------------------
        elif self.stage == 2:
            gripper = 1.0
            target = hand.copy()

            if self.steps >= self.motion.grasp_close_steps:
                self.grasp_position = hand.copy()
                self.grasp_can_height = float(can[2])
                self.grasp_distance = float(distance)
                self.locked_rotation = hand_rot

                contact = self.contact_grasp()

                if contact is False:
                    self.retry(
                        "No grasp contact after closing"
                    )
                else:
                    self.change_stage(
                        3, "Testing grasp by lifting"
                    )

        # -------------------------------
        # 3: Lift and verify
        # -------------------------------
        elif self.stage == 3:
            gripper = 1.0

            target = (
                self.grasp_position
                + np.array([0.0, 0.0, 0.18])
            )

            hand_rise = (
                hand[2] - self.grasp_position[2]
            )

            can_rise = (
                can[2] - self.grasp_can_height
            )

            contact = self.contact_grasp()

            # Wait for the lift to become observable.
            if (
                hand_rise > 0.08
                and can_rise > 0.05
                and distance < 0.14
                and contact is not False
            ):
                self.change_stage(
                    5, "Lift verified; transporting"
                )

            elif self.steps > 160:
                self.retry(
                    "Lift verification failed"
                )

        # -------------------------------
        # 4: Human returned holding can
        # -------------------------------
        elif self.stage == 4:
            gripper = 1.0

            if self.lift_start is None:
                self.lift_start = (
                    hand.copy(), can.copy()
                )

            start_hand, start_can = self.lift_start

            target = start_hand + np.array([
                0.0, 0.0, 0.12
            ])

            hand_rise = hand[2] - start_hand[2]
            can_rise = can[2] - start_can[2]

            contact = self.contact_grasp()

            if (
                hand_rise > 0.05
                and can_rise > 0.03
                and contact is not False
            ):
                self.locked_rotation = hand_rot
                self.lift_start = None
                self.change_stage(
                    5, "Human grasp verified"
                )

            elif self.steps > 140:
                self.lift_start = None
                self.retry(
                    "Human grasp not verified"
                )

        # -------------------------------
        # 5: Transport toward the bin
        # -------------------------------
        elif self.stage == 5:
            gripper = 1.0

            final_target = bin_pos + np.array([0.0, 0.0, 0.20])
            contact = self.contact_grasp()

            if contact is False or distance > 0.18:
                self.retry("Can lost during transport")
            else:
                offset = np.asarray(self.motion.transport_offset_xy)
                if (
                    np.any(offset)
                    and not self.recovery_mode
                    and not self.transport_waypoint_reached
                ):
                    # Calculate ONCE, not every frame: otherwise target moves.
                    if self.transport_waypoint is None:
                        midpoint = (hand[:2] + final_target[:2]) / 2
                        self.transport_waypoint = np.array([
                            midpoint[0] + offset[0],
                            midpoint[1] + offset[1],
                            max(hand[2], final_target[2]) + 0.025,
                        ])
                    target = self.transport_waypoint
                    if np.linalg.norm(target - hand) < 0.040:
                        self.transport_waypoint_reached = True
                    elif self.steps >= self.stage5_waypoint_timeout:
                        print("MOTION: transport waypoint timeout -> direct transport")
                        self.transport_waypoint_reached = True
                else:
                    target = final_target
                    xy_error = np.linalg.norm((target - hand)[:2])
                    if xy_error < 0.04:
                        self.change_stage(6, "Reached bin horizontally")
                    elif self.steps % 100 == 0:
                        print(f"TRANSPORT: XY error={xy_error:.3f}")

        # -------------------------------
        # 6: Lower into the bin
        # -------------------------------
        elif self.stage == 6:
            gripper = 1.0

            target = bin_pos + np.array([
                0.0, 0.0, 0.08
            ])

            contact = self.contact_grasp()

            xy_error = np.linalg.norm(
                (target - hand)[:2]
            )
            z_error = abs(target[2] - hand[2])

            if contact is False and distance > 0.14:
                self.retry("Can lost before release")

            elif xy_error < 0.04 and z_error < 0.06:
                self.change_stage(
                    7, "Releasing can"
                )

            elif self.steps % 100 == 0:
                print(
                    f"LOWERING: XY={xy_error:.3f}, "
                    f"Z={z_error:.3f}"
                )

        # -------------------------------
        # 7: Open gripper
        # -------------------------------
        elif self.stage == 7:
            gripper = -1.0
            target = hand.copy()

            if self.steps >= 40:
                self.change_stage(
                    8, "Retreating"
                )

        # -------------------------------
        # 8: Retreat
        # -------------------------------
        elif self.stage == 8:
            gripper = -1.0

            target = bin_pos + np.array([
                0.0, 0.0, 0.26
            ])

            if self.steps == 1:
                self.home_rotation = hand_rot

        # -------------------------------
        # OSC_POSE command
        # -------------------------------
        position_error = target - hand

        # Change speed only during clear-space travel stages.
        travel_gain = (
            self.motion.transit_gain
            if self.stage in (0, 5) and not self.recovery_mode
            else 5.0
        )
        action[:3] = np.clip(
            position_error * travel_gain,
            -0.5,
            0.5
        )

        action[3:6] = np.clip(
            rotation_error * 2.0,
            -0.15,
            0.15
        )

        action[6] = gripper

        if self.debug_count % 100 == 0:
            print(
                f"STAGE={self.stage} "
                f"TILT={tilt:.1f} "
                f"DIST={distance:.3f} "
                f"CONTACT={self.contact_grasp()}"
            )

        return action
