package config

import (
	"strings"
	"sync"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"
)

var (
	instance *Config
	once     sync.Once
)

// Configuration return Config structure
func configuration() *Config {
	viper.SetConfigName("config") // name of config file (without extension)
	viper.SetConfigType("json")
	viper.AddConfigPath("./config/")
	viper.AddConfigPath(".")
	replacer := strings.NewReplacer(".", "_")
	viper.SetEnvKeyReplacer(replacer)
	addKeysToViper()
	viper.AutomaticEnv()
	if e := viper.ReadInConfig(); e != nil {
		log.Fatalf("Can't read config file %v", e)
	}
	var config Config
	if e := viper.Unmarshal(&config); e != nil {
		log.Fatalf("Can't decode config file to struct %v", e)
	}
	return &config
}

// GetInstance - return Configuration instance
func GetInstance() *Config {
	once.Do(func() {
		instance = configuration()
	})
	return instance
}
