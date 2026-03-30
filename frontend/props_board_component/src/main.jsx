import React from "react";
import ReactDOM from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import { App } from "./app.jsx";

const ConnectedBoard = withStreamlitConnection(App);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ConnectedBoard />
  </React.StrictMode>,
);

Streamlit.setFrameHeight(760);
