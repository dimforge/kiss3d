#!/bin/bash

npm run build:all
cp .htaccess build/.
rsync -av --delete-after build/ crozet@ssh.cluster003.hosting.ovh.net:/home/crozet/kiss3d/
