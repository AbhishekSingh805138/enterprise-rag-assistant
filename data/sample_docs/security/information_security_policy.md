# Acme Corp Information Security Policy

## Purpose
This policy defines the minimum security requirements for all systems, networks,
and data managed by Acme Corp. Compliance is mandatory for all employees,
contractors, and third-party vendors with access to company resources.

## Password Policy
- Minimum length: 14 characters
- Must include at least one uppercase letter, one lowercase letter, one digit,
  and one special character
- Passwords expire every 90 days; the last 12 passwords cannot be reused
- Multi-factor authentication (MFA) is required for all accounts. Approved MFA
  methods: hardware security keys (preferred), authenticator apps (TOTP). SMS-based
  MFA is not permitted.
- Account lockout after 5 consecutive failed login attempts; automatic unlock after 30 minutes.

## Access Control
Access follows the principle of least privilege. Employees are granted access only
to systems and data required for their role. Access reviews are conducted quarterly
by department managers. Privileged access (admin, root, database) requires:
1. Approval from the security team
2. Justification documented in the access management system
3. Time-limited access (maximum 8 hours per session) with automatic revocation
4. Full audit logging of all privileged actions

Terminated employees must have all access revoked within 4 hours of separation.
Contractors' access is automatically revoked at contract end date.

## Endpoint Security
All company-managed devices must have:
- Endpoint Detection and Response (EDR) software (CrowdStrike)
- Full disk encryption (FileVault for macOS, BitLocker for Windows)
- Automatic OS and software updates enabled (maximum 7-day patch window)
- Company-managed antivirus with real-time scanning
- Screen lock after 5 minutes of inactivity

Personal devices (BYOD) are not permitted to access Confidential or Restricted data.
BYOD devices accessing Internal data must have MDM enrolled.

## Network Security
- All internal services must communicate over TLS 1.2 or higher
- VPN is required for remote access to internal networks (WireGuard or Zscaler)
- Network segmentation: production, staging, corporate, and guest networks are isolated
- Outbound traffic is monitored and filtered; connections to known malicious IPs
  are blocked automatically
- Wireless networks use WPA3-Enterprise with certificate-based authentication

## Vulnerability Management
- External-facing systems are scanned weekly for vulnerabilities
- Internal systems are scanned monthly
- Critical vulnerabilities (CVSS 9.0+) must be patched within 48 hours
- High vulnerabilities (CVSS 7.0–8.9) must be patched within 7 days
- Medium vulnerabilities (CVSS 4.0–6.9) must be patched within 30 days
- Annual penetration tests are conducted by an independent third party

## Acceptable Use
Company systems and networks are for business use. Limited personal use is
permitted but must not interfere with job duties or violate any policy.
Prohibited activities include:
- Installing unauthorized software
- Bypassing security controls
- Accessing or distributing inappropriate content
- Using company resources for personal commercial purposes
- Sharing credentials or using shared accounts

## Incident Reporting
All security incidents or suspected incidents must be reported to the Security
Operations Center (SOC) at security@acmecorp.com or via the #security-incidents
Slack channel within 1 hour of discovery. Do not attempt to investigate or
remediate on your own without SOC guidance.
