package main

import (
	"data-collector/internal/config"
	"data-collector/internal/jira"
	"encoding/json"
	"os"

	"github.com/sirupsen/logrus"
)

func main() {
	logrus.SetLevel(logrus.TraceLevel)
	cfg := config.GetInstance()
	jcli := jira.GetInstance(cfg.Jira)
	issues, err := jcli.GetAllTasks()
	if err != nil {
		panic(err)
	}

	file, err := os.Create("data.jsonl")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	for _, issue := range issues {
		err := encoder.Encode(issue)
		if err != nil {
			panic(err)
		}
	}
}
