# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import logging
import traceback
from datetime import datetime, timezone


class LogstashFormatter(logging.Formatter):
    # The list contains all the attributes listed in
    # http://docs.python.org/library/logging.html#logrecord-attributes
    SKIP_ATTRIBUTES = (
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "id",
        "levelname",
        "module",
        "msecs",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "relativeCreated",
        "extra",
        "auth_token",
        "password",
        "stack_info",
    )
    EASY_TYPES = (str, bool, dict, float, int, list, type(None))

    def __init__(self, message_type="Logstash", tags: list[str] | None = None):
        self.message_type = message_type
        self.tags = tags or []

    def get_extra_fields(self, record):
        fields = {}
        for key, value in record.__dict__.items():
            if key not in self.SKIP_ATTRIBUTES:
                if isinstance(value, self.EASY_TYPES):
                    fields[key] = value
                else:
                    fields[key] = repr(value)

        return fields

    def get_debug_fields(self, record):
        fields = {
            "stack_trace": self.format_exception(record.exc_info),
        }
        return fields

    @classmethod
    def format_source(cls, message_type, host, path):
        return f"{message_type}://{host}/{path}"

    @classmethod
    def format_timestamp(cls, time):
        tstamp = datetime.fromtimestamp(time, timezone.utc)
        return (
            tstamp.strftime("%Y-%m-%dT%H:%M:%S")
            + ".{:03d}".format(int(tstamp.microsecond / 1000))
            + "Z"
        )

    @classmethod
    def format_exception(cls, exc_info):
        return "".join(traceback.format_exception(*exc_info)) if exc_info else ""

    @classmethod
    def serialize(cls, message):
        return bytes(json.dumps(message, default=str), "utf-8")

    def format(self, record):
        # Create message dict
        message = {
            "timestamp": self.format_timestamp(record.created),
            "version": "1",
            "message": record.getMessage(),
            "path": record.pathname,
            "tags": self.tags,
            "type": self.message_type,
            # Extra Fields
            "level": record.levelname,
            "logger_name": record.name,
        }

        # Add extra fields
        message.update(self.get_extra_fields(record))
        if record.exc_info:
            message.update(self.get_debug_fields(record))

        return json.dumps(message)
