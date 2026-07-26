import unittest

from src.agents.common.query_expansion import expand_legal_query


class QueryExpansionTests(unittest.TestCase):
    def test_expands_copy_customer_data_into_legal_concepts(self):
        expanded, terms = expand_legal_query(
            "Nhân viên copy danh sách khách hàng sang USB để dùng riêng"
        )
        self.assertIn("dữ liệu cá nhân", expanded)
        self.assertIn("bí mật kinh doanh", terms)
        self.assertIn("nghĩa vụ bảo mật", expanded)

    def test_keeps_unmatched_query_unchanged(self):
        query = "Thời hạn kháng cáo là bao lâu?"
        expanded, terms = expand_legal_query(query)
        self.assertEqual(query, expanded)
        self.assertEqual([], terms)

    def test_sql_injection_expands_to_cybersecurity_not_privacy(self):
        expanded, terms = expand_legal_query(
            "SQL Injection, khai thác lỗ hổng website, tải xuống cơ sở dữ liệu"
        )

        self.assertIn("xâm nhập trái phép hệ thống thông tin", terms)
        self.assertIn("khai thác điểm yếu lỗ hổng bảo mật", expanded)
        self.assertNotIn("sự đồng ý của chủ thể dữ liệu", terms)
        self.assertNotIn("xử lý dữ liệu cá nhân", terms)

    def test_generic_database_does_not_imply_personal_data(self):
        query = "Doanh nghiệp xây dựng và khai thác cơ sở dữ liệu nội bộ"
        expanded, terms = expand_legal_query(query)

        self.assertEqual(query, expanded)
        self.assertEqual([], terms)


if __name__ == "__main__":
    unittest.main()
