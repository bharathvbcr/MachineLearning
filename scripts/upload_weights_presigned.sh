#!/usr/bin/env bash
# PUT the selected final.pt straight from this box to S3, using presigned URLs.
#
# No AWS credentials are installed here and none should be: this instance is
# rented and gets destroyed. Each URL is scoped to one key, one verb, and
# expires in 12 h, so the worst case of the list leaking is bounded by what the
# list already contains. The file is mode 600 and deleted at the end.
#
# Direct box->S3 rather than box->laptop->S3: 25.8 GB once instead of twice, and
# the transfer does not depend on a laptop staying awake.
set -u
LIST="${LIST:-/home/ubuntu/.presigned.tsv}"
[ -r "$LIST" ] || { echo "no URL list at $LIST"; exit 1; }
cd ~/MLSystemsLab/nanolab/out || exit 1

ok=0; bad=0; skipped=0; bytes=0
while IFS=$'\t' read -r path size key url; do
  [ -z "${path:-}" ] && continue
  if [ ! -f "$path" ]; then
    echo "MISSING  $path"; bad=$((bad+1)); continue
  fi
  actual=$(stat -c %s "$path")
  if [ "$actual" != "$size" ]; then
    # The file changed under us since the list was built. Refuse rather than
    # upload bytes the manifest does not describe.
    echo "SIZE-DRIFT $path ($actual != $size)"; bad=$((bad+1)); continue
  fi
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X PUT \
           -H 'Content-Type: application/octet-stream' \
           --retry 3 --retry-delay 5 --retry-connrefused \
           -T "$path" "$url")
  if [ "$code" = "200" ]; then
    ok=$((ok+1)); bytes=$((bytes+actual))
    printf 'ok   %-72s %6.2f GB\n' "$key" "$(echo "$actual" | awk '{print $1/1073741824}')"
  else
    bad=$((bad+1)); echo "HTTP $code  $key"
  fi
done < "$LIST"

printf '\nuploaded %d, failed %d, skipped %d, %.1f GB\n' \
  "$ok" "$bad" "$skipped" "$(echo "$bytes" | awk '{print $1/1073741824}')"
shred -u "$LIST" 2>/dev/null || rm -f "$LIST"
echo "URL list removed from this box"
echo "upload_weights exit=$bad $(date -u +%FT%TZ)"
