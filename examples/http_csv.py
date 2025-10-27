# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# this is a port of the example at
# https://github.com/apache/datafusion/blob/45.0.0/datafusion-examples/examples/query-http-csv.rs

import ray

from datafusion_ray import DFRayContext, df_ray_runtime_env
import time


def main():
    ctx = DFRayContext()
    # ctx.register_csv(
    #     "aggregate_test_100",
    #     "https://github.com/apache/arrow-testing/raw/master/data/csv/aggregate_test_100.csv",
    # )

    ctx.register_parquet(
        "hits",
        "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_1.parquet",
    )
    start = time.perf_counter()
    df = ctx.sql('SELECT SUM(hits."Age") FROM hits')
    df.show()
    end = time.perf_counter()
    print(f"time taken: {end - start} secs")


if __name__ == __name__:
    ray.init(
        namespace="http_csv", runtime_env=df_ray_runtime_env, include_dashboard=False
    )
    main()
