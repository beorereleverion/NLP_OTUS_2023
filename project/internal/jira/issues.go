package jira

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/andygrunwald/go-jira"
	"github.com/sirupsen/logrus"
	"golang.org/x/time/rate"
)

const (
	mr                  = 500
	rpc                 = 30
	maxTimePerRequest   = 90 * time.Second
	requestRetryCounter = 5
)

var (
	limiter = rate.NewLimiter(rate.Limit(rpc), 1)
)

func (i *Instance) GetTaskWithCustomFieldsRateLimmited(taskID string, ch chan<- *Issue) {
	t, err := i.GetTaskWithCustomFields(taskID)
	if err != nil {
		logrus.Fatalf("Failed to get task %s: %s\n", taskID, err)
	}
	ch <- t
}

func (i *Instance) GetTaskWithCustomFields(taskID string) (*Issue, error) {
	var (
		err error
		t   *jira.Issue
	)
	for y := 1; y <= requestRetryCounter; y++ {
		t, _, err = i.Get(taskID, &jira.GetQueryOptions{
			Fields: strings.Join([]string{
				"description",
				"project",
				"summary",
				strings.Join(i.CustomFields, ","),
			}, ","),
		})
		if err == nil {
			break
		}
		logrus.Errorf("can't retrieve task %v, try number %v, err: %v", taskID, y, err)
	}
	if err != nil {
		return nil, err
	}
	customFields := make(map[string]string)
	for _, cfj := range i.CustomFields {
		cf := t.Fields.Unknowns[cfj]
		customFields[cfj] = cf.([]interface{})[0].(string)
	}
	fmt.Printf("%#v\n", Issue{
		Summary:      t.Fields.Summary,
		Description:  t.Fields.Description,
		Project:      t.Fields.Project.Key,
		CustomFields: customFields,
	})
	return &Issue{
		Summary:      t.Fields.Summary,
		Description:  t.Fields.Description,
		Project:      t.Fields.Project.Key,
		CustomFields: customFields,
	}, nil
}

const allProjectTaskPrefix = "project in (%s)"
const cfNotEmptyJQL = " and %s is not EMPTY"

func getAllProjectTaskJQL(projects, customfields []string) string {
	projectsOneStr := strings.Join(projects, ",")
	prefixCompleted := fmt.Sprintf(allProjectTaskPrefix, projectsOneStr)
	var cfs string
	for _, cf := range customfields {
		cfs += fmt.Sprintf(cfNotEmptyJQL, cf)
	}
	return prefixCompleted + cfs
}

func (i *Instance) GetAllTasks() ([]*Issue, error) {
	var (
		tasks         []*Issue
		notEmbedTasks []jira.Issue
	)
	taskCh := make(chan *Issue)
	doneCh := make(chan bool)
	for c := 0; ; c += mr {
		tasks, _, err := i.Search(getAllProjectTaskJQL(i.Projects, i.CustomFieldsJQL), &jira.SearchOptions{
			StartAt:    c,
			MaxResults: mr,
			Fields:     []string{"id"},
		})
		if err != nil {
			return nil, err
		}
		if len(tasks) == 0 {
			break
		}
		notEmbedTasks = append(notEmbedTasks, tasks...)
		logrus.Tracef("time %v maxCount %v\n", time.Now(), c)
	}

	go func() {
		for ti := 0; ti < len(notEmbedTasks); ti++ {
			t := <-taskCh
			if t != nil {
				tasks = append(tasks, t)
			}
		}
		close(doneCh)
	}()
	for number, task := range notEmbedTasks {
		logrus.Tracef("time %v taskkey %v tasknumber %v\n", time.Now(), task.Key, number)
		ctx, cancel := context.WithTimeout(context.Background(), maxTimePerRequest)
		defer cancel()
		if err := limiter.Wait(ctx); err != nil {
			logrus.Fatalf("Failed to rate limit task %s\n", err)
			taskCh <- nil
		}
		go i.GetTaskWithCustomFieldsRateLimmited(task.ID, taskCh)
	}
	<-doneCh
	logrus.Tracef("len of not embed tasks %d", len(notEmbedTasks))
	logrus.Tracef("len of embed tasks %d", len(tasks))
	close(taskCh)
	return tasks, nil
}
