#!/usr/bin/env bash
# Serve site/ headless and check that index.html and every asset in manifest.json answer 200.
cd /home/ec2-user/workspace/depth3d-lab/site || exit 1
PORT=${PORT:-8765}
python3 -m http.server $PORT --bind 127.0.0.1 >/dev/null 2>&1 &
PID=$!
sleep 1
fail=0
for f in $(python3 -c "import json; print('\n'.join(json.load(open('manifest.json')).keys()))"); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/$f")
  if [ "$code" != "200" ]; then echo "FAIL $code $f"; fail=1; fi
done
kill $PID
if [ $fail = 0 ]; then echo "SITE_OK: all $(python3 -c "import json; print(len(json.load(open('manifest.json'))))") files served with 200"; fi
exit $fail
