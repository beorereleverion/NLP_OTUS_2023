package config

// Config root structure. Contain spec of config.json
type Config struct {
	Jira Jira
}

type Jira struct{
	URL                string
	User               string
	Password           string
	Projects           []string
	CustomFieldsNumber []int
}