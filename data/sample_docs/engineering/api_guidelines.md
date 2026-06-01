# Acme Corp API Design Guidelines

## General Principles
All public and internal APIs must follow RESTful design principles. Use nouns for
resource names (e.g., `/users`, `/orders`) and HTTP methods for actions (GET, POST,
PUT, PATCH, DELETE). APIs must be versioned using URL path versioning (e.g., `/v1/users`).
Breaking changes require a new major version.

## Authentication and Authorization
All API endpoints must require authentication. The standard authentication mechanism
is OAuth 2.0 with JWT bearer tokens. Tokens expire after 1 hour; refresh tokens
expire after 30 days. Service-to-service calls use client credentials flow with
mTLS. API keys are permitted only for public read-only endpoints and must be
rotated every 90 days.

## Rate Limiting
Default rate limits per API tier:
- **Public APIs**: 100 requests per minute per API key
- **Partner APIs**: 500 requests per minute per client
- **Internal APIs**: 2,000 requests per minute per service

Rate limit headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`)
must be included in all responses. Exceeding limits returns HTTP 429 with a
`Retry-After` header.

## Request and Response Format
- Use JSON for all request and response bodies.
- Use camelCase for field names.
- Dates must be in ISO 8601 format (e.g., `2026-01-15T09:30:00Z`).
- Pagination: use cursor-based pagination with `cursor` and `limit` parameters.
  Default page size is 20; maximum is 100.
- All list endpoints must support filtering via query parameters and sorting via
  `sort` and `order` parameters.

## Error Handling
Error responses must follow the standard error schema:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable description",
    "details": [{"field": "email", "issue": "invalid format"}]
  }
}
```
Standard HTTP status codes: 400 (validation), 401 (unauthenticated), 403 (forbidden),
404 (not found), 409 (conflict), 422 (unprocessable), 429 (rate limit), 500 (server error).

## Monitoring and Logging
All APIs must emit structured logs (JSON) with request ID, method, path, status code,
latency, and caller identity. P95 latency must be under 200ms for read endpoints
and under 500ms for write endpoints. Health check endpoints (`/health` and `/ready`)
are required for all services.

## Deprecation Policy
Deprecated API versions must be supported for a minimum of 6 months after the
announcement. Deprecation notices must be communicated via the `Deprecation` and
`Sunset` HTTP headers, API changelog, and direct notification to registered consumers.
