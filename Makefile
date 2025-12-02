.PHONY: start
start:
	touch ./data/zeek_logs/conn.log
	touch ./data/ids_metrics_log.jsonl
	touch ./data/logs/ids_api.log
	docker compose up -d --build

.PHONY: stop
stop:
	docker compose down