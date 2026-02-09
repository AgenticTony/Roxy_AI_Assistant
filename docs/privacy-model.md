# Roxy Privacy Model

## Overview

Roxy is designed with privacy as a core principle. This document explains how Roxy protects your data, when and why data might leave your device, and how you maintain control over your information.

## Core Privacy Principles

### 1. Local-First Processing

**Default behavior:** Roxy processes 80%+ of requests entirely on your device using local LLMs (Ollama).

**What stays local:**
- User input and conversation history
- File contents (unless explicitly requested)
- System information
- Memory and preferences
- Voice audio (processed by local STT)

**Benefits:**
- No data transmission to external servers
- Faster response times (<2s vs 5-10s for cloud)
- Works offline
- No API costs

### 2. Privacy-Gated Cloud Access

When local processing isn't sufficient, Roxy may escalate to cloud LLMs—but only with explicit safeguards.

**Cloud escalation triggers:**
- Complex reasoning beyond local model capability
- Requests for recent information (news, current events)
- Specialized tasks requiring specific models
- Low confidence scores from local model

**Privacy protections:**
- PII redaction before sending
- User consent required (configurable)
- Audit logging of all cloud requests
- Minimal data transmission

### 3. User Control

You have full control over Roxy's privacy behavior:

**Consent modes:**
- `never`: Block all cloud LLM usage
- `ask`: Prompt before each cloud request (default)
- `always`: Auto-approve cloud requests

**Configuration:**
```bash
# Environment variable
export ROXY_CLOUD_CONSENT_MODE=ask

# Or in config/default.yaml
privacy:
  cloud_consent: "ask"
```

## Data Flow

### Local Processing Flow

```
User Input
    ↓
[Intent Classification] ← Local (Qwen3 0.6B)
    ↓
[Confidence Scoring] ← Local (Qwen3 0.6B)
    ↓
High confidence (≥0.7)?
    ↓ Yes
[Local LLM Response] ← Local (Qwen3 8B)
    ↓
Response to User
```

**Data stays on device throughout.**

### Cloud Processing Flow

```
User Input
    ↓
[Intent Classification] ← Local (Qwen3 0.6B)
    ↓
[Confidence Scoring] ← Local (Qwen3 0.6B)
    ↓
Low confidence (<0.7)?
    ↓ Yes
[Privacy Gateway]
    ↓
[PII Detection] ← Pattern matching
    ↓
[PII Redaction] ← Replace with placeholders
    ↓
[Consent Check] ← User confirmation (if mode=ask)
    ↓
[Cloud LLM Request] ← Redacted data only
    ↓
[Response Processing]
    ↓
[Log to Audit Trail]
    ↓
Response to User
```

**Only redacted data leaves the device.**

## PII Detection and Redaction

### Detected PII Types

| Type        | Pattern Examples                          | Redacted As         |
|-------------|-------------------------------------------|---------------------|
| Email       | user@example.com                          | `[EMAIL_REDACTED]`  |
| Phone       | 555-123-4567, +1 555 123 4567             | `[PHONE_REDACTED]`  |
| SSN         | 123-45-6789                               | `[SSN_REDACTED]`    |
| Credit Card | 4532-1234-5678-9010                       | `[CC_REDACTED]`     |
| Address     | 123 Main St, Springfield, IL 62701       | `[ADDRESS_REDACTED]`|

### Configuration

Enable/disable patterns in `config/privacy.yaml`:

```yaml
privacy:
  redact_patterns:
    - email
    - phone
    - ssn
    - credit_card
    - address
    - ip_address
    - url

  pii_redaction_enabled: true
```

### Custom Patterns

Add custom PII patterns:

```yaml
privacy:
  custom_patterns:
    - name: "api_key"
      regex: "(?i)api[_-]?key['\"]?\\s*[:=]\\s*['\"]?([a-zA-Z0-9_]+)"
      replace_with: "[API_KEY_REDACTED]"
```

## Audit Logging

### Cloud Request Logs

All cloud requests are logged to `data/cloud_requests.log`:

```
2024-01-15 10:23:45 | CLOUD_REQUEST | provider=zai | confidence=0.5
─────────────────────────────────────────────────────────────────
PROMPT (REDACTED):
"Search for [EMAIL_REDACTED] in [PHONE_REDACTED] area"

RESPONSE:
"Found 3 results..."

─────────────────────────────────────────────────────────────────
```

### Log Contents

Each log entry includes:
- Timestamp
- Provider used (zai, openrouter, etc.)
- Confidence score
- Redacted prompt
- Response summary
- Processing time

### Accessing Logs

```bash
# View recent cloud requests
tail -f data/cloud_requests.log

# Count cloud requests
grep "CLOUD_REQUEST" data/cloud_requests.log | wc -l

# Search for specific patterns
grep "zai" data/cloud_requests.log
```

### Deleting Logs

```bash
# Clear audit log
rm data/cloud_requests.log

# Or anonymize existing logs
sed -i.bak 's/\[.*\]_REDACTED/[REDACTED]/g' data/cloud_requests.log
```

## What Roxy NEVER Sends to Cloud

### Explicitly Excluded

- **File contents** (unless explicitly requested: "summarize this file using cloud")
- **Passwords and API keys** (detected and redacted)
- **Health information** (detected and redacted)
- **Financial information** (detected and redacted)
- **Full conversation history** (only current context)
- **Memory contents** (local-only)
- **System files** (never transmitted)

### Example Data Flow

**User says:** "Send an email to john@example.com about the meeting at 2pm"

**What happens:**
1. Intent detected: Email sending
2. Skill dispatch: EmailSkill
3. Privacy check: Email detected
4. Redaction: `john@example.com` → `[EMAIL_REDACTED]`
5. Cloud request: "Send email to [EMAIL_REDACTED] about meeting at 2pm"
6. Local resolution: Email sent with actual address

**What cloud sees:** Redacted email address only
**What stays local:** Actual email, file attachments, etc.

## Consent Modes

### Mode: `never` (Most Private)

**Behavior:**
- All cloud LLM requests blocked
- Local processing only
- Some features may not work

**Use when:**
- Maximum privacy required
- Offline usage
- Testing local capabilities

```bash
export ROXY_CLOUD_CONSENT_MODE=never
```

### Mode: `ask` (Default - Balanced)

**Behavior:**
- Prompts before each cloud request
- Shows what will be sent (redacted)
- User approves or denies

**Example prompt:**
```
[Roxy] I need to use the cloud for this request.

Request: "What's the latest news about AI?"
Redacted: No PII detected
Provider: Z.ai (ChatGLM)

Allow cloud usage? [y/N]:
```

```bash
export ROXY_CLOUD_CONSENT_MODE=ask
```

### Mode: `always` (Least Private)

**Behavior:**
- Auto-approves all cloud requests
- Still applies PII redaction
- Still logs all requests

**Use when:**
- Trust the privacy gateway
- Need faster responses
- Non-sensitive work

```bash
export ROXY_CLOUD_CONSENT_MODE=always
```

## Memory Privacy

### Session Memory

- **Location:** In-memory only
- **Persistence:** Cleared on shutdown
- **Cloud exposure:** Never sent

### Conversation History

- **Location:** SQLite database (`data/memory.db`)
- **Persistence:** Permanent
- **Cloud exposure:** Never sent
- **Encryption:** Optional (future enhancement)

### Long-term Memory

- **Location:** Mem0 storage (`data/mem0/`)
- **Persistence:** Permanent
- **Cloud exposure:** Never sent
- **User control:** Can be edited/deleted

## Voice Privacy

### Wake Word Detection

- **Processing:** Local (OpenWakeWord)
- **Audio storage:** None
- **Cloud exposure:** Never

### Speech-to-Text (STT)

- **Processing:** Local (faster-whisper)
- **Audio storage:** None (unless debug enabled)
- **Cloud exposure:** Text only, after local processing

### Text-to-Speech (TTS)

- **Processing:** Local (Kokoro/MLX-Audio)
- **Audio storage:** Cached temporarily
- **Cloud exposure:** Never

## Network Traffic

### Local LLM (Ollama)

- **Destination:** localhost:11434
- **External traffic:** None
- **Encryption:** Not needed (local)

### Cloud LLM (Z.ai, OpenRouter)

- **Destination:** api.z.ai, openrouter.ai
- **Encryption:** TLS 1.3
- **Data sent:** Redacted prompts only
- **Authentication:** API key in header

### Web Search (Brave Search)

- **Destination:** api.search.brave.com
- **Encryption:** TLS 1.3
- **Data sent:** Search query (redacted)
- **Authentication:** API key in header

## Security Best Practices

### For Users

1. **Review consent mode:** Set to `ask` for awareness
2. **Check audit logs:** Regularly review `data/cloud_requests.log`
3. **Use strong API keys:** Don't reuse credentials
4. **Enable PII redaction:** Keep `pii_redaction_enabled=true`
5. **Secure your device:** Use macOS FileVault for disk encryption

### For Developers

1. **Declare permissions:** Always declare skill permissions
2. **Use privacy gateway:** For skills that contact cloud APIs
3. **Test with redaction:** Verify PII is properly redacted
4. **Log appropriately:** Don't log sensitive information
5. **Follow principle of least privilege:** Request minimal access

## Transparency

### What Roxy Tells You

1. **Before cloud requests:** "I need to use the cloud..."
2. **In response:** "[Response generated using cloud LLM]"
3. **In logs:** Full audit trail
4. **In stats:** `/stats` shows cloud usage percentage

### How to Check Your Privacy Status

```bash
# In Roxy REPL
/stats

# Shows:
# - Total requests
# - Local vs cloud percentage
# - Recent cloud requests
```

## Privacy Policy Summary

**Data Collection:**
- Roxy collects minimal data for functionality
- No telemetry or usage analytics (by default)
- No user tracking or profiling

**Data Sharing:**
- Cloud LLM providers: Redacted prompts only
- No data sold to third parties
- No advertising or marketing use

**Data Storage:**
- All data stored locally on your Mac
- You control retention and deletion
- Export/backup capabilities available

**User Rights:**
- Right to know: What data is stored
- Right to access: View your data
- Right to delete: Remove your data
- Right to opt-out: Disable cloud features

## Compliance

Roxy's privacy model is designed with consideration for:

- **GDPR:** User control, right to deletion, data minimization
- **CCPA:** Privacy by design, no sale of data
- **SOC 2:** Audit logging, access controls (future)

## Reporting Issues

If you find a privacy or security issue:

1. **Check logs:** Review `data/cloud_requests.log`
2. **Verify settings:** Check consent mode and redaction patterns
3. **Report:** GitHub issues at https://github.com/anthonyforan/roxy

## Future Enhancements

Planned privacy improvements:

- [ ] End-to-end encryption for cloud requests
- [ ] Local-only mode installer option
- [ ] Per-skill consent controls
- [ ] PII detection improvements (ML-based)
- [ ] Data export/privacy dashboard
- [ ] Differential privacy for analytics (opt-in)

## References

- [GDPR Compliance](https://gdpr.eu/)
- [CCPA Compliance](https://oag.ca.gov/privacy/ccpa)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)
