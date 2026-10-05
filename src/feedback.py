"""Private feedback delivery for the Interactive Image Processor."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


class FeedbackConfigurationError(ValueError):
    """Raised when the configured feedback endpoint is not a Formspree form."""


class FeedbackDeliveryError(RuntimeError):
    """Raised when a feedback submission cannot be delivered."""


@dataclass(frozen=True)
class FeedbackSubmission:
    """A private feedback message plus useful, non-personal app context."""

    category: str
    message: str
    technique: str
    rating: int | None = None
    highlights: tuple[str, ...] = ()

    def as_form_fields(self) -> dict[str, str]:
        """Return the fields expected by the hosted feedback form."""

        fields = {
            "_subject": f"Interactive Image Processor: {self.category}",
            "feedback_type": self.category,
            "message": self.message,
            "current_technique": self.technique,
            "source": "Interactive Image Processor",
        }
        if self.rating is not None:
            fields["rating"] = f"{self.rating + 1} / 5"
        if self.highlights:
            fields["highlights"] = ", ".join(self.highlights)
        if not self.message:
            fields["message"] = "No written comment"
        return fields


def validate_formspree_endpoint(endpoint: str) -> str:
    """Validate and normalize a Formspree form endpoint."""

    normalized = endpoint.strip()
    parsed = urlparse(normalized)
    path_parts = parsed.path.strip("/").split("/")
    is_form_endpoint = (
        parsed.scheme == "https"
        and parsed.hostname == "formspree.io"
        and len(path_parts) == 2
        and path_parts[0] == "f"
        and bool(path_parts[1])
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )
    if not is_form_endpoint:
        raise FeedbackConfigurationError(
            "FORMSPREE_ENDPOINT must look like https://formspree.io/f/your-form-id."
        )
    return normalized


def submit_feedback(
    endpoint: str,
    submission: FeedbackSubmission,
    *,
    timeout: float = 10.0,
) -> None:
    """Send one feedback submission without exposing the recipient's email."""

    validated_endpoint = validate_formspree_endpoint(endpoint)
    body = urlencode(submission.as_form_fields()).encode("utf-8")
    request = Request(
        validated_endpoint,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Kasra-Sadatsharifi-Interactive-Image-Processor/1.0",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
    except HTTPError as error:
        raise FeedbackDeliveryError(
            f"The feedback service returned status {error.code}."
        ) from error
    except (TimeoutError, URLError, OSError) as error:
        raise FeedbackDeliveryError(
            "The feedback service could not be reached."
        ) from error

    if not 200 <= status < 300:
        raise FeedbackDeliveryError(f"The feedback service returned status {status}.")
