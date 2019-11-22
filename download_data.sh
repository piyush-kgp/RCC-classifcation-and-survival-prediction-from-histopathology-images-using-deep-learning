
# curl -O https://bootstrap.pypa.io/get-pip.py
# python get-pip.py --user
# python -m pip install virtualenv
# python -m virtualenv v2env
source v2env/bin/activate
# pip install -r requirements_gdc_client.txt

# This should work now.
gdc-client --version

# At this point you should have manifest at ~/manifests
python process_manifests.py

mkdir -p /ssd_scratch/cvit/piyush
cd /ssd_scratch/cvit/piyush
