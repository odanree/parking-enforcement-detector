# ADR 013 — Alert Notification Cascade (Email → ntfy → TextBelt → Twilio)

## Context

The "Send Alert" button in the event modal needs to deliver a real-time notification to a phone. Several SMS/notification providers were evaluated:

- **Twilio 10DLC long code**: blocked by A2P 10DLC registration requirement for US numbers; registration takes weeks.
- **Twilio Toll-Free**: requires organization verification, adding friction for personal/small deployments.
- **TextBelt free tier** (`key=textbelt`): free US SMS disabled due to abuse.
- **TextBelt paid**: works, but requires purchasing credits.
- **ntfy.sh**: free push notifications, but requires installing the ntfy app on the phone.
- **Email**: zero carrier friction, works on any phone via the phone's native email push, uses only standard library (`smtplib`).

## Decision

Implement a provider cascade checked in priority order:

1. **Email** (`ALERT_EMAIL` + `SMTP_USER` + `SMTP_PASS`) — preferred for personal use; Gmail App Passwords work reliably with no registration.
2. **ntfy.sh** (`NTFY_TOPIC`) — free push notifications for users willing to install the ntfy app.
3. **TextBelt** (`TEXTBELT_KEY`) — paid SMS for deployments that need actual SMS delivery.
4. **Twilio** (`TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN` + `TWILIO_FROM_NUMBER`) — fallback for organizations with A2P registration complete.

If none are configured, the endpoint returns HTTP 503 with a clear configuration message.

## Consequences

- No single provider is hardcoded; operators configure only what they have available.
- Email requires a Gmail App Password (2FA must be enabled on the Google account); regular passwords are rejected by Gmail's SMTP.
- The cascade checks in priority order on every request — there is no runtime provider selection UI.
- `httpx` (already a dependency via `anthropic`) handles the async HTTP calls for ntfy and TextBelt; Twilio uses its own SDK.
