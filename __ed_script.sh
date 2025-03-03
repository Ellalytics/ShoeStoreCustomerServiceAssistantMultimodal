#!/bin/bash

cd '/usercode'

sh -c "sed -i '/^export[[:blank:]]GOOGLE_API_KEY=/d' ~/.bashrc && echo 'export GOOGLE_API_KEY= AIzaSyAa9TehprVIT90prnO-aoaogOLCugnqSIo
' >> ~/.bashrc" >> '/usercode/__ed_stdout.txt' 2>> '/usercode/__ed_stderr.txt'
exit 0