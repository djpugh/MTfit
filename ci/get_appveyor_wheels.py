from __future__ import unicode_literals, print_function

import os.path

import requests

api_url = 'https://ci.appveyor.com/api'
account_name = os.getenv('APPVEYOR_ACCOUNT')
project_slug = os.getenv('APPVEYOR_SLUG')
headers = {'Authorization': 'Bearer ' + os.getenv('APPVEYOR_TOKEN')}


# Download the artifacts to wheelhouse/
build_info_url = '{}/projects/{}/{}/build/{}'.format(api_url, account_name, project_slug, os.getenv('TRAVIS_COMMIT'))
r = requests.get(build_info_url)
r.raise_for_status()
build = r.json()
job_ids = [job['jobId'] for job in build['jobs']]
os.mkdir('dist')
for job_id in job_ids:
    r = requests.get('{}/buildjobs/{}/artifacts'.format(api_url, job_id), headers=headers)
    r.raise_for_status()
    for artifact in r.json():
        url = '{}/buildjobs/{}/artifacts/{}'.format(api_url, job_id, artifact['fileName'])
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        file_name = 'dist/' + os.path.basename(artifact['fileName'])
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(None):
                f.write(chunk)
print('Downloaded ' + f.name)
