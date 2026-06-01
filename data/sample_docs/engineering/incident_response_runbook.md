# Acme Corp Engineering Incident Response Runbook

## Severity Levels
- **SEV-1 (Critical)**: Complete service outage or data breach affecting customers.
  Response time: 15 minutes. All hands on deck. Executive notification required.
- **SEV-2 (Major)**: Significant degradation affecting >25% of users or a critical
  business function. Response time: 30 minutes. On-call team plus relevant leads.
- **SEV-3 (Minor)**: Localized issue affecting <25% of users with a workaround
  available. Response time: 2 hours. On-call engineer handles.
- **SEV-4 (Low)**: Cosmetic or non-impacting issue. Handled during business hours.

## Incident Commander Role
Every SEV-1 and SEV-2 incident must have a designated Incident Commander (IC).
The IC is responsible for:
1. Declaring the incident and setting severity.
2. Opening a dedicated Slack channel (#inc-YYYYMMDD-short-description).
3. Assembling the response team and assigning roles (communications, technical lead, scribe).
4. Coordinating troubleshooting and communication.
5. Declaring resolution and scheduling the post-mortem.

The IC does NOT troubleshoot directly — their job is coordination and decision-making.

## Communication Protocol
- **Internal**: Updates every 30 minutes in the incident Slack channel. Status page
  updated within 15 minutes of incident declaration.
- **Customer-facing**: Initial acknowledgment within 30 minutes. Updates every hour
  or when status changes. Final resolution notice within 2 hours of resolution.
- **Executive**: SEV-1 triggers an immediate page to the VP of Engineering and CTO.
  Written summary within 1 hour.

## Post-Mortem Process
A blameless post-mortem is required for all SEV-1 and SEV-2 incidents within 5
business days of resolution. The post-mortem document must include:
- Timeline of events (discovery, response, mitigation, resolution)
- Root cause analysis (using the "5 Whys" technique)
- Impact assessment (users affected, revenue impact, data implications)
- Action items with owners and deadlines
- Lessons learned

Post-mortems are stored in Confluence under the "Incident Post-Mortems" space
and reviewed in the monthly engineering all-hands.

## Rollback Procedures
If a deployment is identified as the cause of an incident:
1. Immediately halt the deployment pipeline.
2. Execute rollback using `deploy rollback --to <previous-version>`.
3. Verify service restoration via health checks and key metrics.
4. Do not attempt a forward fix during a SEV-1; always rollback first, fix later.

Rollback must be completable within 10 minutes for all Tier 1 services.

## On-Call Rotation
On-call rotations are weekly, Monday 9 AM to Monday 9 AM. Each team maintains a
primary and secondary on-call. Primary responds first; secondary is backup if
primary is unreachable within 10 minutes. On-call engineers receive a $500 weekly
stipend. Swaps must be arranged at least 24 hours in advance via PagerDuty.
