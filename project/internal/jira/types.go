package jira

type Issue struct {
	Summary      string            `json:"summary"`
	Description  string            `json:"description"`
	Project      string            `json:"project"`
	CustomFields map[string]string `json:"customFields"`
}
