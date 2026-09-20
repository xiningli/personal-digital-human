import test from "node:test";
import assert from "node:assert/strict";
import { designSpace, pickLine, samplePair, SITE_DEFAULT } from "../lib/policies.ts";

test("the design space holds the site's default and distinct ids", () => {
  const space = designSpace();
  assert.ok(space.some((p) => p.id === SITE_DEFAULT.id));
  assert.equal(new Set(space.map((p) => p.id)).size, space.length);
  assert.ok(space.every((p) => p.source === "clip" || p.source === "generated"));
});

test("a sampled pair is two different policies and favours the unserved", () => {
  const [a, b] = samplePair(new Map());
  assert.notEqual(a.id, b.id);
  const served = new Map(designSpace().map((p) => [p.id, 50]));
  served.set(SITE_DEFAULT.id, 0);
  let hits = 0;
  for (let i = 0; i < 200; i++) { const [x, y] = samplePair(served); if (x.id === SITE_DEFAULT.id || y.id === SITE_DEFAULT.id) hits++; }
  assert.ok(hits > 100, `unserved policy drawn ${hits}/200`);
});

test("a line comes from the requested category", () => {
  assert.equal(pickLine("greeting").category, "greeting");
  assert.ok(pickLine("any").text.length > 10);
});
