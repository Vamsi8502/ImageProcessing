import google.auth

creds, project = google.auth.default()
print(f"✅ Authenticated as: {creds.service_account_email}")
print(f"✅ Project: {project}")
