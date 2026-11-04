#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  build_conditioned_a2_image.sh \
    --project PROJECT \
    --confirm-project PROJECT \
    --region REGION \
    --repository REPOSITORY \
    --image-name IMAGE_NAME \
    --image-tag IMAGE_TAG \
    [--dry-run]

Builds the conditioned-A2 image from the strict upload allowlist and resolves
the resulting tag to an immutable Artifact Registry digest. The two project
arguments must match exactly. --dry-run prints the bounded command only.
EOF
}

project=""
confirm_project=""
region=""
repository=""
image_name=""
image_tag=""
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      project="${2:-}"
      shift 2
      ;;
    --confirm-project)
      confirm_project="${2:-}"
      shift 2
      ;;
    --region)
      region="${2:-}"
      shift 2
      ;;
    --repository)
      repository="${2:-}"
      shift 2
      ;;
    --image-name)
      image_name="${2:-}"
      shift 2
      ;;
    --image-tag)
      image_tag="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

for required_value in project confirm_project region repository image_name image_tag; do
  if [[ -z "${!required_value}" ]]; then
    echo "Missing required value: $required_value" >&2
    usage
    exit 2
  fi
done

if [[ "$project" != "$confirm_project" ]]; then
  echo "Project confirmation mismatch: '$project' != '$confirm_project'" >&2
  exit 2
fi

if [[ ! "$project" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]]; then
  echo "Invalid Google Cloud project ID: $project" >&2
  exit 2
fi

if [[ ! "$region" =~ ^[a-z0-9]+(-[a-z0-9]+)+$ ]]; then
  echo "Invalid Google Cloud region: $region" >&2
  exit 2
fi

for image_part_name in repository image_name image_tag; do
  image_part="${!image_part_name}"
  if [[ ! "$image_part" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid $image_part_name: $image_part" >&2
    exit 2
  fi
done

cloud_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trainer_root="$(cd "$cloud_dir/.." && pwd)"
config="$cloud_dir/cloudbuild-conditioned-a2.yaml"
ignore_file="$cloud_dir/conditioned-a2.gcloudignore"
dockerfile="$cloud_dir/Dockerfile.conditioned-a2"
image_uri="$region-docker.pkg.dev/$project/$repository/$image_name:$image_tag"

for required_file in "$config" "$ignore_file" "$dockerfile"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required build input is missing: $required_file" >&2
    exit 1
  fi
done

build_command=(
  gcloud builds submit "$trainer_root"
  "--project=$project"
  "--region=$region"
  "--config=$config"
  "--ignore-file=$ignore_file"
  "--substitutions=_IMAGE=$image_uri"
)

if [[ "$dry_run" == true ]]; then
  printf 'image=%s\n' "$image_uri"
  printf 'command='
  printf '%q ' "${build_command[@]}"
  printf '\n'
  exit 0
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required" >&2
  exit 1
fi

"${build_command[@]}"

digest="$(
  gcloud artifacts docker images describe "$image_uri" \
    "--project=$project" \
    --format='value(image_summary.digest)'
)"

if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Build completed but the immutable image digest was not resolved" >&2
  exit 1
fi

printf 'image_digest_uri=%s@%s\n' "${image_uri%:*}" "$digest"
