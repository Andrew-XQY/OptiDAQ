"""Offline motor-raster lifecycle checks; no CERN SDK or hardware required."""

from copy import deepcopy
import json
import unittest

from optidaq.actuators.motor_point_raster import (
    MotorPointRasterController, make_motor_raster_plan, validate_motor_raster_config,
)


def config():
    return {
        "motors": {
            "position_mode": "raw_counter", "return_to_start": True,
            "x": {"device": "test.x", "enabled": True, "poll_s": 0.01,
                  "timeout_s": 1.0, "tol_counts": 5, "settle_s": 0.0},
            "y": {"device": "test.y", "enabled": True, "poll_s": 0.01,
                  "timeout_s": 1.0, "tol_counts": 5, "settle_s": 0.0},
        },
        "scan": {"nx": 10, "ny": 10, "serpentine": True,
                 "stride_counts": {"x": 100, "y": 100},
                 "start_offset_counts": {"x": -400, "y": -400}},
    }


def span_config():
    cfg = config()
    cfg["scan"].pop("stride_counts")
    cfg["scan"].pop("start_offset_counts")
    cfg["scan"]["span_counts"] = {"x": 10000, "y": 10000}
    return cfg


class FakeAxis:
    def __init__(self, name, client, cfg, *, require_valid_position):
        self.name, self.device = name, cfg["device"]
        self.counter = {"x": 79912, "y": 19908}[name]
        self.units, self.units_scale, self.units_offset = "Steps", 1.0, 0.0
        self.position_min, self.position_max = -1000000, 1000000
        self.tol_counts, self.settle_s = cfg["tol_counts"], cfg["settle_s"]
        self.timeout_s, self.poll_s = cfg["timeout_s"], cfg["poll_s"]
        self.require_valid_position = require_valid_position
        self.busy, self.valid = False, False
        self.channel_status, self.driver_error = 1032, 0
        self.moves, self.stops = [], 0
        self.position_bias, self.counter_error = 0, 0
        self.on_move = None
        self.fail_move = False

    def acquisition(self):
        return {"position_counter": self.counter, "position": self.counter+self.position_bias,
                "position_valid": self.valid, "motor_busy": self.busy,
                "status_channel": self.channel_status,
                "end_switch_minus_active": False, "end_switch_plus_active": False}

    def driver_acquisition(self):
        return {"ps_power_state": True, "ps_error": self.driver_error}

    def preflight(self):
        acq = self.acquisition()
        if acq["motor_busy"]:
            raise RuntimeError("already busy")
        if self.require_valid_position and not acq["position_valid"]:
            raise RuntimeError("position invalid")
        self.start_counter, self.start_position = self.counter, acq["position"]

    def _counter(self, acq):
        return acq["position_counter"]

    def _position(self, acq):
        return acq["position"]

    def position_for_counter(self, counter):
        return counter * self.units_scale - self.units_offset

    def range_check_absolute_counters(self, values):
        if any(not self.position_min <= self.position_for_counter(value) <= self.position_max for value in values):
            raise RuntimeError("outside limits")

    def move_abs_raw(self, target):
        self.acquisition()
        if abs(self.counter - target) <= self.tol_counts:
            return
        self.moves.append(target)
        self.busy = True
        if self.on_move:
            self.on_move()
        # Emulate MotorAxis polling its acquisition callable while moving.
        self.acquisition()
        if self.fail_move:
            raise RuntimeError("move timeout")
        self.counter = target+self.counter_error
        self.busy = False
        self.acquisition()

    def stop(self):
        self.stops += 1
        self.busy = False


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration


class MotorPointRasterTests(unittest.TestCase):
    def controller(self, cfg=None, setup=None, should_stop=lambda: False, clock=None):
        axes = {}

        def factory(*args, **kwargs):
            axis = FakeAxis(*args, **kwargs)
            if setup:
                setup(axis)
            axes[axis.name] = axis
            return axis

        controller = MotorPointRasterController(None, cfg or config(), should_stop,
                                                axis_factory=factory, clock=clock or FakeClock())
        return controller, axes

    def test_hundred_unique_points_serpentine_adjacency(self):
        plan = make_motor_raster_plan(config())
        self.assertEqual(len(plan), 100)
        self.assertEqual(len({(p.x_offset_counts, p.y_offset_counts) for p in plan}), 100)
        self.assertEqual(plan[0].to_dict(), {
            "index": 0, "row": 0, "column": 0,
            "x_offset_counts": -400, "y_offset_counts": -400,
        })
        self.assertEqual((plan[9].column, plan[10].column), (9, 9))
        self.assertEqual((plan[-1].x_offset_counts, plan[-1].y_offset_counts), (-400, 500))
        for first, second in zip(plan, plan[1:]):
            self.assertEqual(abs(second.x_offset_counts-first.x_offset_counts)
                             + abs(second.y_offset_counts-first.y_offset_counts), 100)

    def test_invalid_geometry_and_tolerances_fail_before_hardware(self):
        cases = [("nx", 0), ("nx", 10.5), ("ny", True), ("serpentine", "yes")]
        for key, value in cases:
            cfg = config()
            cfg["scan"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                make_motor_raster_plan(cfg)
        for step in (0, 10, 100.1, True):
            cfg = config()
            cfg["scan"]["stride_counts"]["x"] = step
            with self.subTest(step=step), self.assertRaises(ValueError):
                validate_motor_raster_config(cfg)
        for tolerance in (-1, float("nan"), 50):
            cfg = config()
            cfg["motors"]["x"]["tol_counts"] = tolerance
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                validate_motor_raster_config(cfg)

    def test_span_has_hundred_unique_symmetric_integer_points_and_exact_endpoints(self):
        plan = make_motor_raster_plan(span_config())
        self.assertEqual(len(plan), 100)
        self.assertEqual(len({(p.x_offset_counts, p.y_offset_counts) for p in plan}), 100)
        for name in ("x", "y"):
            offsets = sorted({getattr(point, name + "_offset_counts") for point in plan})
            self.assertEqual(offsets, [-5000, -3889, -2778, -1667, -556,
                                      556, 1667, 2778, 3889, 5000])
            self.assertTrue(all(type(value) is int for value in offsets))
            self.assertEqual(offsets, [-value for value in reversed(offsets)])
            self.assertEqual({b - a for a, b in zip(offsets, offsets[1:])}, {1111, 1112})
        for first, second in zip(plan, plan[1:]):
            distances = (abs(second.x_offset_counts - first.x_offset_counts),
                         abs(second.y_offset_counts - first.y_offset_counts))
            self.assertIn(0, distances)
            self.assertIn(sum(distances), (1111, 1112))

    def test_span_rejects_ambiguous_or_unresolvable_integer_geometry(self):
        for key in ("stride_counts", "start_offset_counts"):
            cfg = span_config()
            cfg["scan"][key] = {"x": 100, "y": 100}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cannot be combined"):
                validate_motor_raster_config(cfg)
        for width in (0, -2, True, 10000.5, 9999, 2):
            cfg = span_config()
            cfg["scan"]["span_counts"]["x"] = width
            with self.subTest(width=width), self.assertRaises(ValueError):
                validate_motor_raster_config(cfg)
        cfg = span_config()
        cfg["scan"]["nx"] = 1
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_motor_raster_config(cfg)

    def test_reference_requires_both_integer_int32_counters(self):
        for reference in (None, {}, {"x": 100}, {"x": 1.5, "y": 2},
                          {"x": True, "y": 2}, {"x": 2**31, "y": 2}):
            cfg = span_config()
            cfg["motors"]["reference_counters"] = reference
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                validate_motor_raster_config(cfg)

    def test_explicit_reference_centers_scan_and_returns_to_distinct_live_start(self):
        cfg = span_config()
        cfg["motors"]["reference_counters"] = {"x": 90000, "y": -1000}
        original = deepcopy(cfg)
        controller, axes = self.controller(cfg)
        controller.preflight()
        controller.prepare()
        self.assertEqual(controller.initial_counters, {"x": 79912, "y": 19908})
        for point in controller.plan:
            info = controller.apply_point(point)
            self.assertEqual(info["reference_counters"], cfg["motors"]["reference_counters"])
            self.assertEqual(info["initial_counters"], controller.initial_counters)
            self.assertEqual(info["origin"], "configured_reference_counters")
            held = controller.verify_held(point)
            for name in ("x", "y"):
                target = cfg["motors"]["reference_counters"][name] + getattr(point, name + "_offset_counts")
                self.assertEqual(info["target_counters"][name], target)
                self.assertEqual(held[name]["position_counter"], target)
        controller.cleanup(success=True)
        self.assertEqual({name: axis.counter for name, axis in axes.items()}, controller.initial_counters)
        saved = controller.actuator_metadata()
        self.assertEqual(saved["reference_counters"], {"x": 90000, "y": -1000})
        self.assertEqual(saved["origin"], "configured_reference_counters")
        self.assertTrue(saved["restored_within_tolerance"])
        self.assertEqual(cfg, original)

    def test_preflight_checks_reference_targets_and_live_return_before_any_move(self):
        for reference_x, minimum, maximum, error in ((95000, 70000, 95000, "limits"),
                                                    (90000, 80000, 100000, "limits"),
                                                    (2**31 - 1, -1e10, 1e10, "int32")):
            cfg = span_config()
            cfg["motors"]["reference_counters"] = {"x": reference_x, "y": 19908}

            def limits(axis):
                if axis.name == "x":
                    axis.position_min, axis.position_max = minimum, maximum

            controller, axes = self.controller(cfg, setup=limits)
            with self.subTest(reference=reference_x, minimum=minimum), self.assertRaisesRegex(RuntimeError, error):
                controller.preflight()
            self.assertTrue(all(not axis.moves for axis in axes.values()))
    def test_full_raster_has_actual_coordinates_and_exact_return(self):
        cfg = config()
        original = deepcopy(cfg)
        controller, axes = self.controller(cfg)
        controller.preflight()
        controller.prepare()
        self.assertEqual(controller.initial_counters, {"x": 79912, "y": 19908})
        for point in controller.plan:
            info = controller.apply_point(point)
            held = controller.verify_held(point)
            for name in ("x", "y"):
                self.assertEqual(info["readbacks"][name]["position_counter"], info["target_counters"][name])
                self.assertEqual(held[name]["position_counter"], info["target_counters"][name])
                self.assertFalse(info["readbacks"][name]["position_valid"])
                self.assertIsInstance(info["readbacks"][name]["acquisition"], dict)
            json.dumps(info, allow_nan=False)
        controller.cleanup(success=True)
        self.assertEqual({name: axis.counter for name, axis in axes.items()}, controller.initial_counters)
        self.assertTrue(controller.restored)
        self.assertEqual(controller.actuator_metadata()["position_mode"], "raw_counter")
        self.assertEqual(controller.actuator_metadata()["reference_counters"], controller.initial_counters)
        self.assertEqual(controller.actuator_metadata()["origin"], "live_preflight_counters")
        self.assertFalse(controller.actuator_metadata()["homing_performed"])
        self.assertEqual(cfg, original)

    def test_entire_range_checked_before_any_move(self):
        def narrow(axis):
            if axis.name == "y":
                axis.position_max = axis.counter+400
        controller, axes = self.controller(setup=narrow)
        with self.assertRaisesRegex(RuntimeError, "limits"):
            controller.preflight()
        controller.cleanup(success=False)
        self.assertTrue(all(not axis.moves for axis in axes.values()))

    def test_hardware_fault_blocks_preflight_without_stop_or_move(self):
        controller, axes = self.controller(setup=lambda axis: setattr(axis, "channel_status", 1))
        with self.assertRaisesRegex(RuntimeError, "hardware problem"):
            controller.preflight()
        controller.cleanup(success=False)
        self.assertTrue(all(not axis.moves and not axis.stops for axis in axes.values()))

    def test_raw_mode_still_requires_counter_calibration_consistency(self):
        controller, _ = self.controller(setup=lambda axis: setattr(axis, "position_bias", 16))
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            controller.preflight()
        cfg = config()
        cfg["motors"]["position_mode"] = "valid_position"
        controller, _ = self.controller(cfg)
        with self.assertRaisesRegex(RuntimeError, "position is not valid"):
            controller.preflight()

    def test_fault_during_move_stops_commanded_axis_and_no_return_move(self):
        controller, axes = self.controller()
        controller.preflight()
        axes["x"].on_move = lambda: setattr(axes["x"], "driver_error", 7)
        with self.assertRaisesRegex(RuntimeError, "driver"):
            controller.apply_point(controller.plan[0])
        controller.cleanup(success=False)
        self.assertEqual(axes["x"].moves, [79512])
        self.assertEqual(axes["x"].stops, 1)
        self.assertEqual(axes["y"].moves, [])
        self.assertFalse(controller.restored)

    def test_cancellation_during_poll_stops_without_advancing_other_axis(self):
        cancelled = [False]
        controller, axes = self.controller(should_stop=lambda: cancelled[0])
        controller.preflight()
        axes["x"].on_move = lambda: cancelled.__setitem__(0, True)
        with self.assertRaises(KeyboardInterrupt):
            controller.apply_point(controller.plan[0])
        controller.cleanup(success=False)
        self.assertEqual(axes["x"].stops, 1)
        self.assertEqual(axes["y"].moves, [])
        self.assertEqual(len(axes["x"].moves), 1)

    def test_counter_shortfall_is_rejected_not_hidden_by_rounding(self):
        controller, axes = self.controller()
        controller.preflight()
        axes["x"].counter_error = 16
        with self.assertRaisesRegex(RuntimeError, "differs from target"):
            controller.apply_point(controller.plan[0])
        controller.cleanup(success=False)
        self.assertEqual(len(axes["x"].moves), 1)

    def test_move_timeout_stops_and_does_not_continue_the_raster(self):
        controller, axes = self.controller()
        controller.preflight()
        axes["x"].fail_move = True
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            controller.apply_point(controller.plan[0])
        controller.cleanup(success=False)
        self.assertEqual(axes["x"].stops, 1)
        self.assertEqual(axes["y"].moves, [])
        self.assertEqual(len(axes["x"].moves), 1)

    def test_failed_stop_is_bounded_logged_and_does_not_hide_move_error(self):
        clock = FakeClock()
        controller, axes = self.controller(clock=clock)
        controller.preflight()
        axes["x"].fail_move = True
        # Match the shared adapter: a failed STOP write can return normally.
        axes["x"].stop = lambda: setattr(axes["x"], "stops", axes["x"].stops + 1)
        with self.assertRaisesRegex(RuntimeError, "move timeout"):
            controller.apply_point(controller.plan[0])
        attempt = controller.actuator_metadata()["abort_attempts"][-1]
        self.assertTrue(attempt["stop_requested"])
        self.assertFalse(attempt["idle_confirmed"])
        self.assertIn("STOP not confirmed", attempt["confirmation_error"])
        self.assertAlmostEqual(clock.elapsed, 1.0)
        with self.assertRaisesRegex(RuntimeError, "STOP not confirmed"):
            controller.cleanup(success=False)
        self.assertEqual(axes["x"].moves, [79512])
        self.assertEqual(axes["y"].moves, [])
        json.dumps(controller.actuator_metadata(), allow_nan=False)

    def test_stop_exception_does_not_prevent_stopping_other_commanded_axis(self):
        clock = FakeClock()
        controller, axes = self.controller(clock=clock)
        controller.preflight()
        controller.apply_point(controller.plan[0])
        for axis in axes.values():
            axis.busy = True

        def failed_stop():
            axes["x"].stops += 1
            raise RuntimeError("STOP network failure")

        axes["x"].stop = failed_stop
        self.assertFalse(controller.abort())
        attempts = controller.actuator_metadata()["abort_attempts"]
        self.assertIn("STOP network failure", attempts[0]["stop_error"])
        self.assertFalse(attempts[0]["idle_confirmed"])
        self.assertTrue(attempts[1]["idle_confirmed"])
        self.assertEqual([axes[name].stops for name in ("x", "y")], [1, 1])
        self.assertAlmostEqual(clock.elapsed, 1.0)

    def test_missing_or_invalid_busy_cannot_confirm_stop(self):
        for busy in (None, 0, "false"):
            controller, axes = self.controller()
            controller.preflight()
            controller.apply_point(controller.plan[0])
            controller._raw_acquisition["x"] = lambda: {"motor_busy": busy}
            with self.subTest(busy=busy):
                self.assertFalse(controller.abort())
                attempt = controller.actuator_metadata()["abort_attempts"][-2]
                self.assertFalse(attempt["idle_confirmed"])
                self.assertIn("missing or invalid", attempt["readback_error"])
                self.assertEqual(axes["x"].stops, 1)

    def test_held_readback_detects_external_motion(self):
        controller, axes = self.controller()
        controller.preflight()
        point = controller.plan[0]
        controller.apply_point(point)
        axes["y"].counter += 10
        with self.assertRaisesRegex(RuntimeError, "differs from target"):
            controller.verify_held(point)
        controller.cleanup(success=False)
        self.assertFalse(controller.restored)

    def test_failed_cleanup_does_not_reposition_and_success_can_leave_last_point(self):
        for success, return_to_start in ((False, True), (True, False)):
            cfg = config()
            cfg["motors"]["return_to_start"] = return_to_start
            controller, axes = self.controller(cfg)
            controller.preflight()
            controller.apply_point(controller.plan[0])
            moves = {name: list(axis.moves) for name, axis in axes.items()}
            controller.cleanup(success=success)
            controller.cleanup(success=success)
            self.assertEqual({name: axis.moves for name, axis in axes.items()}, moves)
            self.assertFalse(controller.restored)


if __name__ == "__main__":
    unittest.main()
