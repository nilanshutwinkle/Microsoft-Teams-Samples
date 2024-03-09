#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

""" Bot Configuration """


class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "53dd95a9-7517-469f-a16e-6ab3e145009f")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "5qR8Q~5U7~DHbtSnakI2GETqqKoMrNumOam2ia15")
