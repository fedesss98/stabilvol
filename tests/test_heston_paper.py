import unittest

import numpy as np
import pandas as pd

from stabilvol.heston.paper_reproduction import (
    count_fortran_hitting_events, count_hitting_events, fortran_mfht_curve,
    market_return_scale, mfht_curve,
)
from stabilvol.utility.classes.stability_analysis import StabilVolter


class PaperReproductionTests(unittest.TestCase):
    def test_crash_and_rally_events_match_existing_counter(self):
        values = np.array([[0.0], [-0.6], [-0.7], [-1.6], [0.0]])
        for series, start, end in ((values, -0.5, -1.5), (-values, 0.5, 1.5)):
            with self.subTest(start=start, end=end):
                fast = count_hitting_events(series, start, end, tau_min=2, tau_max=30)
                analyst = StabilVolter(
                    start_level=start, end_level=end, std_normalization=False,
                    tau_min=2, tau_max=30,
                )
                analyst.data = pd.DataFrame(series, index=pd.date_range("2000-01-01", periods=5))
                slow = analyst.count_stock_fht(analyst.data.iloc[:, 0])
                self.assertEqual(len(fast), 1)
                self.assertEqual(int(fast.iloc[0]["FHT"]), int(slow[1, 0]))
                self.assertAlmostEqual(fast.iloc[0]["Volatility"], float(slow[0, 0]))

    def test_endpoint_choice_is_explicit(self):
        values = np.array([[0.0], [-0.6], [-0.7], [-1.6]])
        excluded = count_hitting_events(values, -0.5, -1.5)
        included = count_hitting_events(values, -0.5, -1.5, include_end_in_volatility=True)
        self.assertEqual(int(excluded.iloc[0]["FHT"]), 3)
        self.assertNotEqual(excluded.iloc[0]["Volatility"], included.iloc[0]["Volatility"])
        self.assertAlmostEqual(excluded.iloc[0]["Volatility"], np.std(values[:3], ddof=1))

    def test_disjoint_events_restart_after_each_crossing(self):
        values = np.array([[0.0], [-0.6], [-1.6], [0.0], [-0.6], [-1.6]])
        events = count_hitting_events(values, -0.5, -1.5)
        self.assertEqual(events["FHT"].tolist(), [2, 2])
        self.assertEqual(events["start_step"].tolist(), [0, 3])
        self.assertEqual(events["end_step"].tolist(), [2, 5])

    def test_market_scale_and_mfht_bins(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        self.assertAlmostEqual(market_return_scale(values), 1.5)
        events = pd.DataFrame({"Volatility": [0.1, 0.2, 0.2], "FHT": [2, 4, 6]})
        curve = mfht_curve(events, bins=3, vol_max=0.3)
        self.assertEqual(int(curve["events"].sum()), 3)
        self.assertAlmostEqual(curve.loc[curve["events"] == 2, "mfht"].iloc[0], 5.0)

    def test_fortran_counter_includes_crossing_and_uses_population_std(self):
        values = np.array([[0.0], [-0.2], [-1.6], [0.0]])
        for direction, series in (("crash", values), ("rally", -values)):
            with self.subTest(direction=direction):
                events = count_fortran_hitting_events(series, 1.0, direction=direction)
                self.assertEqual(events["FHT"].tolist(), [2])
                self.assertEqual(events["volatility_observations"].tolist(), [2])
                self.assertAlmostEqual(float(events.iloc[0]["Volatility"]), np.std([-0.2, -1.6], ddof=0))

    def test_fortran_invalid_event_carries_state(self):
        values = np.array([[0.0], [-0.2], [-1.6], [0.0], [-0.2], [-0.3], [-1.6]])
        events = count_fortran_hitting_events(values, 1.0, tau_min=3)
        self.assertEqual(events["FHT"].tolist(), [5])
        self.assertEqual(events["volatility_observations"].tolist(), [5])
        self.assertAlmostEqual(float(events.iloc[0]["Volatility"]), np.std([-.2, -1.6, -.2, -.3, -1.6]))

    def test_fortran_reported_axis_is_half_physical_axis(self):
        events = pd.DataFrame({"Volatility": [0.01234], "FHT": [90]})
        curve = fortran_mfht_curve(events)
        row = curve.loc[curve["events"] == 1].iloc[0]
        self.assertAlmostEqual(row["volatility"], 0.00618)
        self.assertLessEqual(row["physical_volatility_lower"], 0.01234)
        self.assertGreater(row["physical_volatility_upper"], 0.01234)


if __name__ == "__main__":
    unittest.main()
