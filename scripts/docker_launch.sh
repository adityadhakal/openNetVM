container_name=$1
if [ -z $1 ] 
then
  container_name="dummy_container"
fi
sudo ./docker.sh -h /mnt/huge -o / -n $container_name -D /dev/uio0,/dev/uio1
#sudo ./docker.sh -h /mnt/huge -o /home/adhak001/dev/openNetVM_sameer -n $container_name -D /dev/uio0,/dev/uio1
