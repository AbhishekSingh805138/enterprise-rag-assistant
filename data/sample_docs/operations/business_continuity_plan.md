# Acme Corp Business Continuity Plan

## Purpose
This plan ensures Acme Corp can maintain essential operations during and after
a significant disruption, including natural disasters, cyberattacks, pandemics,
and infrastructure failures.

## Recovery Objectives
- **Recovery Time Objective (RTO)**: Critical systems must be restored within 4 hours.
  Non-critical systems within 24 hours.
- **Recovery Point Objective (RPO)**: Maximum acceptable data loss is 1 hour for
  critical systems (achieved through continuous replication) and 24 hours for
  non-critical systems (daily backups).

## Critical Business Functions
The following functions are classified as critical and prioritized for recovery:
1. Customer-facing SaaS platform (Tier 1)
2. Payment processing and billing systems (Tier 1)
3. Customer support and ticketing (Tier 1)
4. Internal communication (email, Slack) (Tier 2)
5. Development and CI/CD infrastructure (Tier 2)
6. HR and payroll systems (Tier 2)
7. Marketing and analytics platforms (Tier 3)

## Backup Strategy
- **Database backups**: Continuous replication to a secondary region (US-East to US-West).
  Daily snapshots retained for 30 days. Monthly snapshots retained for 1 year.
- **Application state**: Infrastructure-as-code (Terraform) stored in version control.
  Container images stored in a geo-replicated registry.
- **Document backups**: Google Workspace data backed up daily to a separate cloud provider.
- **Backup testing**: Full restoration test conducted quarterly. Results documented
  and shared with the executive team.

## Disaster Recovery Procedures
### Datacenter/Cloud Region Failure
1. DNS failover triggers automatically (Route 53 health checks, 60-second TTL).
2. Secondary region assumes production traffic.
3. On-call SRE team validates failover within 15 minutes.
4. Customer communication sent within 30 minutes.
5. Post-failover: root cause analysis within 48 hours.

### Ransomware/Cyberattack
1. Isolate affected systems immediately (network segmentation).
2. Activate incident response team and external forensics firm (Mandiant, under retainer).
3. Restore from last known clean backup after forensic preservation.
4. Do NOT pay ransom — company policy prohibits ransom payments.
5. Notify law enforcement (FBI IC3) and affected customers per breach notification requirements.

### Pandemic/Workforce Disruption
1. Activate full remote work for all employees.
2. Ensure VPN capacity supports 100% remote workforce.
3. Cross-training ensures minimum 2 people can perform each critical function.
4. Temporary staffing agreements are pre-arranged with 3 staffing agencies.

## Communication Plan
- **Internal**: Automated alerts via PagerDuty, Slack #emergency channel, and email.
  Phone tree activated for SEV-1 if digital channels are unavailable.
- **External**: Status page (status.acmecorp.com) updated within 15 minutes.
  Customer email notification within 1 hour. Press/media handled exclusively by
  the Communications team.

## Testing Schedule
- **Tabletop exercise**: Quarterly (simulated scenario, discussion-based)
- **Functional drill**: Semi-annually (actual failover of one system)
- **Full DR test**: Annually (complete region failover with production traffic)

## Plan Ownership
The VP of Operations owns this plan and is responsible for annual review and updates.
The last review was completed on January 15, 2026. Next scheduled review: July 15, 2026.
