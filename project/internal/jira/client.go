package jira

import (
	"data-collector/internal/config"
	"fmt"
	"sync"

	gojira "github.com/andygrunwald/go-jira"
	"github.com/sirupsen/logrus"
)

var (
	instance *Instance
	once     sync.Once
)

type Instance struct {
	issueClient
	Projects        []string
	CustomFields    []string
	CustomFieldsJQL []string
}

type issueClient interface {
	Get(issueID string, options *gojira.GetQueryOptions) (*gojira.Issue, *gojira.Response, error)
	Search(jql string, options *gojira.SearchOptions) ([]gojira.Issue, *gojira.Response, error)
}

func GetInstance(cfg config.Jira) *Instance {
	once.Do(func() {
		tp := gojira.CookieAuthTransport{
			Username: cfg.User,
			Password: cfg.Password,
			AuthURL:  cfg.URL + "rest/auth/1/session",
		}
		client, err := gojira.NewClient(tp.Client(), cfg.URL)
		if err != nil {
			logrus.Fatalf("jira can't initialize new client,err: %v", err)
		}
		instance = &Instance{
			issueClient:     client.Issue,
			Projects:        cfg.Projects,
			CustomFields:    getCustomFields(cfg.CustomFieldsNumber, customFieldFormat),
			CustomFieldsJQL: getCustomFields(cfg.CustomFieldsNumber, customFieldJQLFormat),
		}
	})
	return instance
}

const (
	customFieldFormat    = "customfield_%d"
	customFieldJQLFormat = "cf[%d]"
)

func getCustomFields(strs []int, format string) []string {
	var res []string
	for _, str := range strs {
		res = append(res, fmt.Sprintf(format, str))
	}
	return res
}
