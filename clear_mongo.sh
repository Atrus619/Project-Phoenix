./start_mongo.sh || true
mongo --eval "use phoenixdb"
mongo --eval "db.post.remove({})"
