

python torch-baseline/run_dpg.py --algo TD3 --env /content/drive/MyDrive/envs/Crawler.x86_64 --steps 1e6 --worker_id 0 --target_update_tau 0.995 --node 512 --hidden_n 3 --logdir drive/MyDrive/logs/ &
python torch-baseline/run_dpg.py --algo TD4_QR --env /content/drive/MyDrive/envs/Crawler.x86_64 --steps 1e6 --worker_id 1 --target_update_tau 0.995 --node 512 --hidden_n 3 --logdir drive/MyDrive/logs/ &
python torch-baseline/run_dpg.py --algo TD4_QR --env /content/drive/MyDrive/envs/Crawler.x86_64 --steps 1e6 --worker_id 2 --target_update_tau 0.995 --node 512 --hidden_n 3 --logdir drive/MyDrive/logs/ --risk_avoidance 1.0 &
python torch-baseline/run_dpg.py --algo TD4_QR --env /content/drive/MyDrive/envs/Crawler.x86_64 --steps 1e6 --worker_id 3 --target_update_tau 0.995 --node 512 --hidden_n 3 --logdir drive/MyDrive/logs/ --risk_avoidance normal &