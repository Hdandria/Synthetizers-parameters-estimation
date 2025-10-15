@echo off
REM Start interactive shell in Docker container

echo Starting interactive shell in Docker container...
echo Use 'exit' to leave the container

REM Run the shell service
docker-compose run --rm shell-service

echo Shell session ended.
