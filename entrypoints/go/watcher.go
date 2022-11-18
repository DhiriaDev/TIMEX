package main

import (
	"fmt"
	"github.com/confluentinc/confluent-kafka-go/kafka"
    "os"
    "os/exec"
)

type Data struct {
    output []byte
    error  error
}

func runCommand(ch chan<- Data, message string) {
    boostrapServers := os.Args[1]
    role := os.Args[2]

    cmd := exec.Command("python", role + ".py", boostrapServers, message)
    data, err := cmd.CombinedOutput()
    fmt.Printf("Data: %s", data)

	if err != nil {
        fmt.Printf("Err: %s", err)
	}

    ch <- Data{
        error:  err,
        output: data,
    }
}

func main() {

    boostrapServers := os.Args[1]
    role := os.Args[2]

	c, err := kafka.NewConsumer(&kafka.ConfigMap{
        "bootstrap.servers": boostrapServers,
		"group.id":          role,
		"auto.offset.reset": "earliest",
	})

	if err != nil {
		panic(err)
	}

	c.SubscribeTopics([]string{"control_topic"}, nil)

	for {
		msg, err := c.ReadMessage(-1)
		if err == nil {
            c := make(chan Data)

            // This will work in background
            go runCommand(c, string(msg.Value))

		} else {
			// The client will automatically try to recover from all errors.
			fmt.Printf("Consumer error: %v (%v)\n", err, msg)
            c.Close()
		}
	}
}
