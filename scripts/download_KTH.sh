# wget -P datasets/KTH/ http://www.nada.kth.se/cvap/actions/walking.zip
# wget -P datasets/KTH/ http://www.nada.kth.se/cvap/actions/jogging.zip
# wget -P datasets/KTH/ http://www.nada.kth.se/cvap/actions/running.zip
# wget -P datasets/KTH/ http://www.nada.kth.se/cvap/actions/boxing.zip
# wget -P datasets/KTH/ http://www.nada.kth.se/cvap/actions/handwaving.zip
# wget -P datasets/KTH/ http://www.nada.kth.se/cvap/actions/handclapping.zip

unzip datasets/KTH/boxing.zip -d datasets/KTH/boxing
unzip datasets/KTH/handclapping.zip -d datasets/KTH/handclapping
unzip datasets/KTH/handwaving.zip -d datasets/KTH/handwaving
unzip datasets/KTH/jogging.zip -d datasets/KTH/jogging
unzip datasets/KTH/running.zip -d datasets/KTH/running
unzip datasets/KTH/walking.zip -d datasets/KTH/walking

# rm datasets/KTH/*.zip
