import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.retrieval.legal_domains import (
    document_legal_domains,
    select_legal_domains,
)


class LegalDomainTests(unittest.TestCase):
    def test_deepfake_advertising_selects_behavior_domains_not_ip(self):
        selection = select_legal_domains(
            "Deepfake AI dùng hình ảnh người nổi tiếng để quảng cáo"
        )

        self.assertIn("cybersecurity", selection.selected)
        self.assertIn("personal_data", selection.selected)
        self.assertIn("civil_personality", selection.selected)
        self.assertIn("advertising", selection.selected)
        self.assertNotIn("intellectual_property", selection.selected)

    def test_ip_requires_explicit_ip_signal(self):
        selection = select_legal_domains(
            "Sao chép tác phẩm và video có bản quyền để kinh doanh"
        )

        self.assertIn("intellectual_property", selection.selected)

    def test_every_current_docx_has_registered_domains(self):
        corpus = Path(__file__).resolve().parents[1] / "data" / "keep"
        sources = sorted(corpus.glob("*.docx"))
        self.assertTrue(sources)
        unregistered = [
            source.name
            for source in sources
            if document_legal_domains(source.name) == ["general_legal"]
        ]
        self.assertEqual([], unregistered)

    def test_qdrant_match_any_filters_nested_legal_domains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = QdrantClient(path=tmpdir)
            client.create_collection(
                collection_name="chunks",
                vectors_config={
                    "dense": models.VectorParams(
                        size=2, distance=models.Distance.COSINE
                    )
                },
            )
            client.upsert(
                collection_name="chunks",
                points=[
                    models.PointStruct(
                        id=1,
                        vector={"dense": [1.0, 0.0]},
                        payload={
                            "metadata": {"legal_domains": ["cybersecurity"]}
                        },
                    ),
                    models.PointStruct(
                        id=2,
                        vector={"dense": [1.0, 0.0]},
                        payload={
                            "metadata": {
                                "legal_domains": ["intellectual_property"]
                            }
                        },
                    ),
                ],
            )
            result = client.query_points(
                collection_name="chunks",
                query=[1.0, 0.0],
                using="dense",
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.legal_domains",
                            match=models.MatchAny(any=["cybersecurity"]),
                        )
                    ]
                ),
                limit=5,
                with_payload=True,
            )
            self.assertEqual([1], [point.id for point in result.points])
            client.close()


if __name__ == "__main__":
    unittest.main()
