echo "Start deploy ml ranger api"

cd ~
cd ML_Rangers
git pull origin master
yes y | sudo docker system prune -a
sudo docker build -t mlapi .
sudo docker-compose up -d 

echo "End deploy ml ranger api"