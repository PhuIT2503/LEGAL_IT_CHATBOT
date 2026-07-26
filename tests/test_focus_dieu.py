import unittest

from src.agents.common.focus_dieu import compute_focus_dieu_ids


class FocusDieuTests(unittest.TestCase):
    def test_uses_true_max_score_not_first_item_score(self):
        selected = compute_focus_dieu_ids(
            ["D1", "D2", "D3"],
            {"D1": 0.2, "D2": 0.9, "D3": 0.4},
            score_ratio=0.6,
            max_dieu=4,
        )

        self.assertEqual(["D2"], selected)


if __name__ == "__main__":
    unittest.main()
