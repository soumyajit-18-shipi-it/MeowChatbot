# Secrets Rotation & Incident Response

This repository previously contained secret API keys in a `.env` file which have been removed from the Git history and `.env` is now ignored.

Immediate actions (you or repo admins must perform):

1. Revoke compromised keys
   - Open the provider consoles (OpenAI, Groq) and immediately revoke/delete the exposed API keys.

2. Create replacement keys
   - Generate new API keys and store them securely (see recommendations below).

3. Update local/deployment environments
   - Replace the old keys with the new ones in your local `.env` (never commit this file).
   - Update CI/CD secrets (GitHub Actions Secrets, GitLab CI/CD variables, Vault, etc.).

4. Verify no remaining secrets in repo
   - Secret scanning should be clean; if your remote blocked pushes earlier, the history-clean step already removed `.env` from commits.

5. Rotate any downstream tokens
   - If keys were used in other services, rotate those credentials as well.

Longer-term recommendations

- Use repository secret stores (GitHub/GitLab secrets) or a secrets manager (HashiCorp Vault, AWS Secrets Manager) for deployments.
- Do not keep provider keys in plaintext files checked into source control. Keep a `.env.example` with placeholder keys (this repo contains one).
- Configure secret scanning and monitoring on your remote to get alerts for future exposures.

Provider docs
- OpenAI: https://platform.openai.com/docs
- Groq: https://groq.com/docs
