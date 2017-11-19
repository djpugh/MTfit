Building a release
******************

* Tag the release with a release message and push to github
* Build the distribution wheels (use vagrant to build the linux wheels) (N.B. be careful with the c files built from cython)
* Build the documentation using::
     tox -e gh_pages
* Add the release info to github
* Push the release to pypi using `twine`::
    twine upload dist/*
