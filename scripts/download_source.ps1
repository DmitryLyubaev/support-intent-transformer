param(
    [Parameter(Mandatory = $true)][uri]$Uri,
    [Parameter(Mandatory = $true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
if ($Uri.Scheme -ne 'https' -or $Uri.Host -ne 'huggingface.co' -or $Uri.UserInfo) {
    throw 'Only the approved HTTPS Hugging Face origin is allowed'
}
Invoke-WebRequest -Uri $Uri -OutFile $OutputPath
