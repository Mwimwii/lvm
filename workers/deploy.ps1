# Deploy script for Cloudflare Worker — reads secrets from token.json and .env
# Run from lvm/ root:  cd workers; .\deploy.ps1

$ErrorActionPreference = "Stop"

$tokenJson = Get-Content ..\token.json | ConvertFrom-Json
$envLines = Get-Content ..\.env

$refreshToken = $tokenJson.refresh_token
$clientId     = $tokenJson.client_id
$clientSecret = $tokenJson.client_secret
$r2AccessKey  = ($envLines | Where-Object { $_ -match "^R2_ACCESS_KEY_ID=" }) -replace "^R2_ACCESS_KEY_ID=", "" -replace "\s+$",""
$r2SecretKey  = ($envLines | Where-Object { $_ -match "^R2_SECRET_ACCESS_KEY=" }) -replace "^R2_SECRET_ACCESS_KEY=", "" -replace "\s+$",""

Write-Host "Setting wrangler secrets (via npx wrangler)..." -ForegroundColor Cyan
Write-Host "  YT_REFRESH_TOKEN = $($refreshToken.Substring(0,8))..."
Write-Host "  YT_CLIENT_ID     = $clientId"
Write-Host "  YT_CLIENT_SECRET = $($clientSecret.Substring(0,8))..."
Write-Host "  R2_ACCESS_KEY_ID = $($r2AccessKey.Substring(0,8))..."
Write-Host "  R2_SECRET_ACCESS_KEY = $($r2SecretKey.Substring(0,8))..."

function Set-Secret($name, $value) {
    $value | npx wrangler secret put $name
}

Set-Secret "YT_REFRESH_TOKEN"    $refreshToken
Set-Secret "YT_CLIENT_ID"        $clientId
Set-Secret "YT_CLIENT_SECRET"    $clientSecret
Set-Secret "R2_ACCESS_KEY_ID"    $r2AccessKey
Set-Secret "R2_SECRET_ACCESS_KEY" $r2SecretKey

Write-Host "Deploying worker..." -ForegroundColor Cyan
npx wrangler deploy

Write-Host "Done. Worker URL will be shown above." -ForegroundColor Green
Write-Host "Add it to ../.env as CF_WORKER_URL=<url>" -ForegroundColor Yellow
