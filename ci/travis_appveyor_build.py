from __future__ import unicode_literals, print_function

import os.path
import sys
import time

import requests


api_url = 'https://ci.appveyor.com/api'
account_name = os.getenv('APPVEYOR_ACCOUNT')
project_slug = os.getenv('APPVEYOR_SLUG')
headers = {'Authorization': 'Bearer ' + os.getenv('APPVEYOR_TOKEN')}


class Log:

    def __init__(self):
        self.job_status = {}
        self._log = {}

    def update_job(self, job_id, job_status):
        current_status = self.job_status.get(job_id, 'queued')
        if job_status != 'queued' and not (current_status == 'success' and job_status == 'success'):
            # We need to get and update the log here
            r = requests.get('{}/buildjobs/{}/log'.format(api_url, job_id), headers=headers)
            r.raise_for_status()
            new_log = r.content.decode()
            log_diff = new_log.replace(self._log.get(job_id, ''), '').lstrip()
            self._log[job_id] = new_log
            print(log_diff.strip('/r').strip())
        elif not (current_status == 'success' and job_status == 'success'):
            print('{} - Build status: {}; checking again in 10 seconds'.format(job_id, job_status))
        self.job_status[job_id] = job_status


# Trigger the AppVeyor build
payload = {
    'accountName': account_name,
    'projectSlug': project_slug,
    'branch': os.getenv('TRAVIS_BRANCH'),
    'commitID': os.getenv('TRAVIS_COMMIT')
}
r = requests.post(api_url + '/builds', payload, headers=headers)
r.raise_for_status()
build = r.json()
print('Started AppVeyor build (buildId={buildId}, version={version})'.format(**build))
log = Log()
# Wait until the build has finished
has_responded = False
while True:
    url = '{}/projects/{}/{}/build/{}'.format(api_url, account_name, project_slug,
                                              build['version'])
    r = requests.get(url, headers=headers)
    if r.status_code == 404 and not has_responded:
        print('Waiting for build ...')
        time.sleep(5)
    else:
        has_responded = True
        r.raise_for_status()
        build = r.json()['build']
        status = build['status']
        if status in ('queued'):
            print('Build status: {}; checking again in 10 seconds'.format(status))
            time.sleep(10)
        elif status in ('running'):
            # Get the logs and update here
            for job in build['jobs']:
                log.update_job(job['jobId'], job['status'])
            time.sleep(30)
        elif status == 'success':
            print('Build successful')
            break
        else:
            print('Build failed with status: {}'.format(status), file=sys.stderr)
            sys.exit(1)
