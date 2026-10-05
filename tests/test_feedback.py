"""Tests for private feedback delivery."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

from src.feedback import (
    FeedbackConfigurationError,
    FeedbackDeliveryError,
    FeedbackSubmission,
    submit_feedback,
    validate_formspree_endpoint,
)


class FeedbackDeliveryTests(unittest.TestCase):
    repository_root = Path(__file__).resolve().parents[1]

    def test_real_secret_file_is_gitignored(self) -> None:
        gitignore = (self.repository_root / ".gitignore").read_text(encoding="utf-8")

        self.assertIn(".streamlit/secrets.toml", gitignore)

    def test_endpoint_must_be_an_https_formspree_form(self) -> None:
        self.assertEqual(
            validate_formspree_endpoint(" https://formspree.io/f/example-id "),
            "https://formspree.io/f/example-id",
        )

        for endpoint in (
            "",
            "http://formspree.io/f/example-id",
            "https://example.com/f/example-id",
            "https://formspree.io/not-a-form",
        ):
            with (
                self.subTest(endpoint=endpoint),
                self.assertRaises(FeedbackConfigurationError),
            ):
                validate_formspree_endpoint(endpoint)

    @patch("src.feedback.urlopen")
    def test_submission_contains_anonymous_structured_feedback(
        self, mock_urlopen: MagicMock
    ) -> None:
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response

        submit_feedback(
            "https://formspree.io/f/example-id",
            FeedbackSubmission(
                category="Very useful",
                message="Add a side-by-side threshold comparison.",
                technique="Thresholding / Otsu threshold",
                rating=4,
                highlights=("Helpful code examples", "Add more techniques"),
            ),
        )

        request = mock_urlopen.call_args.args[0]
        fields = parse_qs(request.data.decode("utf-8"))
        self.assertEqual(
            request.headers["User-agent"],
            "Kasra-Sadatsharifi-Interactive-Image-Processor/1.0",
        )
        self.assertEqual(fields["source"], ["Interactive Image Processor"])
        self.assertEqual(fields["feedback_type"], ["Very useful"])
        self.assertEqual(fields["current_technique"], ["Thresholding / Otsu threshold"])
        self.assertEqual(fields["rating"], ["5 / 5"])
        self.assertEqual(
            fields["highlights"],
            ["Helpful code examples, Add more techniques"],
        )
        self.assertNotIn("email", fields)
        self.assertNotIn("recipient", fields)

    def test_quick_rating_does_not_require_a_written_comment(self) -> None:
        fields = FeedbackSubmission(
            category="Useful",
            message="",
            technique="Edges / Canny edges",
            rating=3,
        ).as_form_fields()

        self.assertEqual(fields["rating"], "4 / 5")
        self.assertEqual(fields["message"], "No written comment")

    @patch("src.feedback.urlopen")
    def test_delivery_failure_is_reported(self, mock_urlopen: MagicMock) -> None:
        response = MagicMock()
        response.status = 503
        mock_urlopen.return_value.__enter__.return_value = response

        with self.assertRaises(FeedbackDeliveryError):
            submit_feedback(
                "https://formspree.io/f/example-id",
                FeedbackSubmission(
                    category="General feedback",
                    message="Useful tool.",
                    technique="Color spaces / HSV controls",
                ),
            )
