docker build -t inference:latest .

docker run --device=/dev/kfd -p 8004:8000 inference:latest
