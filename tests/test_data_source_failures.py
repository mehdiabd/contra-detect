from unittest import TestCase
from unittest.mock import Mock, patch

import requests

from src.data import elastic
from src.data.errors import DataSourceError


class DataSourceFailureTests(TestCase):
    def test_transport_and_http_errors_become_source_errors(self):
        for error in (requests.ConnectionError("offline"), requests.Timeout("timeout")):
            with self.subTest(error=error), patch("src.data.elastic.requests.request", side_effect=error):
                with self.assertRaises(DataSourceError):
                    elastic._request("POST", "/test/_search")
        for status in (400, 401, 404, 503):
            response = Mock(status_code=status)
            response.raise_for_status.side_effect = requests.HTTPError(response=response)
            with self.subTest(status=status), patch("src.data.elastic.requests.request", return_value=response):
                with self.assertRaises(DataSourceError) as raised:
                    elastic._request("POST", "/test/_search")
                self.assertEqual(raised.exception.status_code, status)

    def test_incomplete_or_invalid_successful_http_response_is_not_empty_data(self):
        for payload in ({"timed_out": True}, {"_shards": {"failed": 1}}, []):
            response = Mock(content=b"json")
            response.json.return_value = payload
            with self.subTest(payload=payload), patch("src.data.elastic.requests.request", return_value=response):
                with self.assertRaises(DataSourceError):
                    elastic._request("POST", "/test/_search")
        response = Mock(content=b"invalid json")
        response.json.side_effect = ValueError("invalid json")
        with patch("src.data.elastic.requests.request", return_value=response):
            with self.assertRaises(DataSourceError):
                elastic._request("POST", "/test/_search")

    def test_partial_fetch_failure_is_not_silently_accepted(self):
        with patch("src.data.elastic.configured_indexes", return_value={"twitter": ["first", "second"]}), \
             patch("src.data.elastic._fetch_from_index", side_effect=[[Mock()], DataSourceError("second source failed")]):
            with self.assertRaises(DataSourceError):
                elastic.fetch_posts(["audit-user"], platforms=["twitter"])

    def test_successful_empty_search_is_valid(self):
        with patch("src.data.elastic.configured_indexes", return_value={"twitter": ["test"]}), \
             patch("src.data.elastic._request", return_value={"hits": {"hits": []}}):
            self.assertEqual(elastic.fetch_posts(["audit-user"], platforms=["twitter"]), [])

    def test_missing_search_results_is_a_source_error(self):
        with patch("src.data.elastic._request", return_value={}):
            with self.assertRaises(DataSourceError):
                elastic._fetch_from_index("twitter", "test", ["audit-user"], 15, 100)

    def test_missing_requested_platform_source_is_an_error(self):
        with patch("src.data.elastic.configured_indexes", return_value={}):
            with self.assertRaises(DataSourceError):
                elastic.fetch_posts(["audit-user"], platforms=["telegram"])
            with self.assertRaises(DataSourceError):
                elastic.fetch_active_users(platforms=["telegram"])

    def test_user_field_mapping_fallback_still_works(self):
        with patch("src.data.elastic.configured_indexes", return_value={"twitter": ["test"]}), \
             patch("src.data.elastic._user_field_candidates", return_value=["unusable", "valid"]), \
             patch("src.data.elastic._request", side_effect=[DataSourceError("field mapping", 400),
                   {"aggregations": {"users": {"buckets": [{"key": "audit-user"}]}}}]):
            self.assertEqual(elastic.fetch_active_users(platforms=["twitter"]), ["audit-user"])

    def test_active_user_query_failure_is_not_hidden_by_other_platform_success(self):
        with patch("src.data.elastic.configured_indexes", return_value={"twitter": ["first"], "telegram": ["second"]}), \
             patch("src.data.elastic._user_field_candidates", return_value=["user_name"]), \
             patch("src.data.elastic._request", side_effect=[
                 {"aggregations": {"users": {"buckets": [{"key": "audit-user"}]}}}, DataSourceError("offline", 503)]):
            with self.assertRaises(DataSourceError):
                elastic.fetch_active_users_by_platform()

    def test_failed_mapping_queries_are_not_an_empty_user_list(self):
        with patch("src.data.elastic.configured_indexes", return_value={"twitter": ["test"]}), \
             patch("src.data.elastic._user_field_candidates", return_value=["bad_field"]), \
             patch("src.data.elastic._request", side_effect=DataSourceError("field mapping", 400)):
            with self.assertRaises(DataSourceError):
                elastic.fetch_active_users(platforms=["twitter"])
