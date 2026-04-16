import React from "react";
import { PropsBoard } from "./props_board.jsx";
import { DataTable } from "./data_table.jsx";
import { GameSelector } from "./game_selector.jsx";
import { ZoneTool } from "./zone_tool.jsx";

export function App({ args }) {
  if (args?.componentType === "data_table") {
    return <DataTable args={args} />;
  }
  if (args?.componentType === "zone_tool") {
    return <ZoneTool args={args} />;
  }
  if (args?.componentType === "game_selector") {
    return <GameSelector args={args} />;
  }
  return <PropsBoard args={args} />;
}
