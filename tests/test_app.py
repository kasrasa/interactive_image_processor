"""Streamlit-level smoke tests for the application interface."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    def test_default_app_renders_without_exceptions(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, "Color spaces")
        self.assertEqual(app.selectbox[1].value, "HSV controls")
        self.assertEqual(app.segmented_control[0].label, "Appearance")
        self.assertEqual(app.segmented_control[0].value, "Light")
        self.assertEqual(app.toggle[0].label, "Grayscale")
        self.assertFalse(app.toggle[0].value)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Learn", "Copy the code", "Compare settings", "Histogram"],
        )
        rendered_markdown = "\n".join(block.value for block in app.markdown)
        normalized_markdown = " ".join(rendered_markdown.split())
        self.assertIn("Created by", rendered_markdown)
        self.assertIn("Interactive Image Processor", rendered_markdown)
        self.assertIn("Kasra Sadatsharifi", rendered_markdown)
        self.assertIn("Research mindset. Production habits.", rendered_markdown)
        self.assertIn(
            "Created as an independent computer-vision learning project by",
            normalized_markdown,
        )
        self.assertIn("https://github.com/kasrasa", rendered_markdown)
        self.assertEqual(len(app.feedback), 1)
        self.assertIsNone(app.feedback[0].value)
        self.assertEqual(app.pills[0].label, "What stood out? (optional)")
        self.assertEqual(app.pills[0].value, [])
        self.assertEqual(app.text_area[0].label, "Anything else I should know?")
        self.assertEqual(app.button[-1].label, "Send anonymous feedback")
        self.assertTrue(
            any(
                "No account, sign-in, or email is required" in caption.value
                for caption in app.caption
            )
        )
        self.assertNotIn("@gmail.com", rendered_markdown)

    def test_empty_feedback_prompts_for_one_quick_input(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.button[-1].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("Choose a face" in warning.value for warning in app.warning)
        )
        rendered_markdown = "\n".join(block.value for block in app.markdown)
        self.assertNotIn("@gmail.com", rendered_markdown)

    def test_configured_delivery_is_required_after_feedback_is_given(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.feedback[0].set_value(4)
        app.button[-1].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("something went wrong" in error.value for error in app.error)
        )

    @patch.dict(
        "os.environ",
        {"FORMSPREE_ENDPOINT": "https://formspree.io/f/example-id"},
    )
    @patch("src.feedback.urlopen")
    def test_feedback_submission_shows_a_private_success_message(
        self, mock_urlopen: MagicMock
    ) -> None:
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response

        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.feedback[0].set_value(4)
        app.pills[0].set_value(["Helpful code examples", "Add more techniques"])
        app.text_area[0].set_value("Please add another threshold comparison.")
        app.button[-1].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any(
                "anonymous feedback was sent privately" in success.value
                for success in app.success
            )
        )
        mock_urlopen.assert_called_once()

        request = mock_urlopen.call_args.args[0]
        request_body = request.data.decode("utf-8")
        self.assertIn("rating=5+%2F+5", request_body)
        self.assertIn("Helpful+code+examples%2C+Add+more+techniques", request_body)

    def test_appearance_control_switches_to_dark_theme(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.segmented_control[0].set_value("Dark").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.segmented_control[0].value, "Dark")
        rendered_markdown = "\n".join(block.value for block in app.markdown)
        self.assertIn("--lab-bg: #0b1120", rendered_markdown)
        self.assertIn("--lab-color-scheme: dark", rendered_markdown)
        self.assertIn('[data-testid="stFeedback"] button', rendered_markdown)
        self.assertIn('button[data-variant="pills"]', rendered_markdown)
        self.assertIn("color: var(--lab-ink) !important", rendered_markdown)

    def test_switching_to_canny_updates_controls_and_code(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.selectbox[0].select("Edges").run()
        app.selectbox[1].select("Canny edges").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            [(slider.label, slider.value) for slider in app.slider],
            [("Low / high thresholds", (60, 150)), ("Pre-blur kernel", 5)],
        )
        self.assertTrue(any("cv2.Canny" in block.value for block in app.code))

    def test_transform_domain_family_renders_fourier_controls_and_code(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.selectbox[0].select("Transform domains").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[1].value, "Fourier spectrum")
        self.assertEqual(app.selectbox[2].value, "Magnitude spectrum")
        self.assertEqual(
            [(checkbox.label, checkbox.value) for checkbox in app.checkbox],
            [("Center zero frequency", True), ("Log magnitude", True)],
        )
        self.assertTrue(any("cv2.dft" in block.value for block in app.code))

    def test_grayscale_toggle_updates_the_working_input_and_code(self) -> None:
        app = AppTest.from_file(self.app_path, default_timeout=30).run()
        app.toggle[0].set_value(True).run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.toggle[0].value)
        self.assertTrue(
            any("Grayscale input" in caption.value for caption in app.caption)
        )
        self.assertTrue(
            any("cv2.COLOR_GRAY2RGB" in block.value for block in app.code)
        )

        app.toggle[0].set_value(False).run()

        self.assertEqual(len(app.exception), 0)
        self.assertFalse(app.toggle[0].value)
        self.assertTrue(any("RGB input" in caption.value for caption in app.caption))
        self.assertFalse(
            any("cv2.COLOR_GRAY2RGB" in block.value for block in app.code)
        )


if __name__ == "__main__":
    unittest.main()
