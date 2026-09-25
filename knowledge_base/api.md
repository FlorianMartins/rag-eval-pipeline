# Developer API

The REST API is versioned and the current version is v3.

Requests are authenticated with a bearer token created in the admin console.

The API rate limit is 600 requests per minute per organization.

When the rate limit is exceeded the API returns HTTP status 429 with a Retry-After header.

Webhooks are signed with HMAC SHA-256 using the secret shown in the admin console.
