from __future__ import unicode_literals, print_function

import os
import re

import requests
REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


api_url = 'https://ci.appveyor.com/api'
account_name = os.getenv('APPVEYOR_ACCOUNT')
project_slug = os.getenv('APPVEYOR_SLUG')
headers = {'Authorization': 'Bearer ' + os.getenv('APPVEYOR_TOKEN')}
travis_headers = {'Authorization': 'token ' + os.getenv('TRAVIS_TOKEN'), 'Travis-API-Version': '3'}

appveyor_version = None

# This is a real messy thing here to get the appveyor version
# we are going to go back over the test stage job logs and look for the line
# Started AppVeyor build (buildId={buildId}, version={version})
# To start with lets get the jobs
travis_api_url = 'https://api.travis-ci.org'
travis_build_stage_url = '{}/build/{}/jobs'.format(travis_api_url, os.getenv('TRAVIS_JOB_ID'))
r = requests.get(travis_build_stage_url, headers=travis_headers)
r.raise_for_status()
jobs = r.json()
job_ids = [u['id'] for u in jobs['jobs'] if u['stage']['name'].lower() == 'test']
# Now we need to get the logs and check if that line is in it
for id_ in job_ids:
    travis_logs_url = '{}/job/{}/log'.format(travis_api_url, id_)
    r = requests.get(travis_logs_url, headers=travis_headers)
    r.raise_for_status()
    # So we try to split on our search string
    log = r.json()['content']
    appveyor_versions = re.findall("Started AppVeyor build \(buildId=.*$", log, re.MULTILINE)
    if appveyor_versions:
        appveyor_version = appveyor_versions[0]
        break

if appveyor_version is None:
    raise RuntimeError('No appveyor build found in test stage')

# Download the artifacts to wheelhouse/
build_info_url = '{}/projects/{}/{}/build/{}'.format(api_url, account_name, project_slug, appveyor_version)
r = requests.get(build_info_url)
r.raise_for_status()
build = r.json()
job_ids = [job['jobId'] for job in build['jobs']]
if not os.path.exists(os.path.join(REPO_PATH, 'dist'))
os.mkdir(os.path.join(REPO_PATH, 'dist'))
for job_id in job_ids:
    r = requests.get('{}/buildjobs/{}/artifacts'.format(api_url, job_id), headers=headers)
    r.raise_for_status()
    for artifact in r.json():
        url = '{}/buildjobs/{}/artifacts/{}'.format(api_url, job_id, artifact['fileName'])
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        file_name = os.path.join(REPO_PATH, 'dist', os.path.basename(artifact['fileName']))
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(None):
                f.write(chunk)
            f.close()
print('Downloaded ' + f.name)
