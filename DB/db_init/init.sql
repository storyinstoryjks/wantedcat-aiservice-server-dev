CREATE DATABASE IF NOT EXISTS my_app_db;
USE my_app_db;

CREATE TABLE IF NOT EXISTS events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255),
    event_time DATETIME,
    duration_seconds FLOAT,
    weight_info VARCHAR(255),
    video_url VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS aimodel (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255),
    model_status VARCHAR(255),
    model_name VARCHAR(255),
    val_precision FLOAT,
    val_recall FLOAT,
    val_map50 FLOAT
);