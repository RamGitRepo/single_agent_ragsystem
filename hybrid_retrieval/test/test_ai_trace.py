import time
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

logging.basicConfig(level=logging.DEBUG)

# Full connection string for Application Insights

# Configure the tracer provider
trace.set_tracer_provider(
    TracerProvider(
        resource=Resource.create({SERVICE_NAME: "test-app-insights", "cloud.role": "test-app-role"})
    )
)

# Exporter and processor setup
exporter = AzureMonitorTraceExporter.from_connection_string(AI_CONNECTION_STRING)
span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Get tracer and create a span
tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("test_connection_span") as span:
    span.set_attribute("test.attribute", "test-value")
    print("Span 'test_connection_span' created and sent to Application Insights")

# ✅ Sleep and shutdown to flush spans before process exits
time.sleep(15)
trace.get_tracer_provider().shutdown()
